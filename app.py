#!/usr/bin/env python3

import asyncio
import logging
import sys
import queue
from time import sleep

import httpx
from asgiref.sync import async_to_sync
from asgiref.wsgi import WsgiToAsgi
from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.httpexceptions import HTTPFound
from pyramid.view import view_config

logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [%(module)s] %(message)s',
)
log = logging.getLogger(__name__)


# log.addHandler(logging.StreamHandler(sys.stdout))


async def list_dog_breeds():
    async with httpx.AsyncClient() as client:
        breeds = (await client.get('https://dog.ceo/api/breeds/list/all')).json()
        log.info(breeds)
        return breeds


# @view_config(
#     route_name='dogstream',
# )
# def dogstream(request):
#     def dog_breeds_iter():
#         breeds = async_to_sync(list_dog_breeds)()
#         breeds_list = breeds['message'].keys()
#         for breed in breeds_list:
#             sleep(1)
#             yield f'data: {breed}\n\n'.encode('utf-8')
#     return Response(
#         app_iter=dog_breeds_iter(),
#         headerlist=[
#             ('Content-Type', 'text/event-stream'),
#             ('Cache-Control', 'no-cache'),
#             ('Connection', 'keep-alive')
#         ]
#     )

async def queue_dog_breeds(q: queue.Queue):
    async with httpx.AsyncClient() as client:
        breeds = (await client.get('https://dog.ceo/api/breeds/list/all')).json()
        breeds_list = breeds['message'].keys()
        log.info(breeds_list)
        for breed in breeds_list:
            log.debug(f'Queueing {breed}')
            q.put_nowait(breed)
            # sleep(1)
            await asyncio.sleep(0.6)
        q.put(None)  # All Done
        log.debug('Dog Breeds Queueing Done')


@view_config(
    route_name='dogstream',
)
def dogstream(request):
    q = queue.Queue()
    # async_to_sync(queue_dog_breeds)(q)
    run_in_background(queue_dog_breeds, q)

    def dog_breeds_iter():
        sentinela = object()
        for breed in iter(q.get, sentinela):
            if breed is None:
                yield f'event: DogBreedNoMass\ndata: No more doggo senior :-(\n\n'.encode('utf-8')
                # break
                return StopIteration
            else:
                yield f'event: DogBreedNoMass\ndata: More doggo senior :-)\n\n'.encode('utf-8')
            yield f'event: DogBreed\ndata: {breed}\n\n'.encode('utf-8')
            # sleep(1)

    return Response(
        app_iter=dog_breeds_iter(),
        headerlist=[
            ('Content-Type', 'text/event-stream'),
            ('Cache-Control', 'no-cache')
        ]
    )

@view_config(
    route_name='doggo',
    renderer='templates/doggo.jinja2'
)
def doggo(request):
    return {}


@view_config(
    route_name='procesar'
)
def procesar(request):
    log.debug(request.POST.items())
    async_to_sync(list_dog_breeds)()
    return HTTPFound(location=request.route_url('home'))


# @view_config(route_name='home')
# def home(request):
#     return Response('Hello World!')

@view_config(
    route_name='home',
    renderer='templates/home.jinja2'
)
def home(request):
    log.info('Welcome to the home page')
    return {'greet': 'Welcome', 'name': 'Djehuti'}


async def persist_openai_chat(query, response):
    log.debug(f'Persisting {query} => {response}')


async def run_openai(q: queue.Queue):
    from time import perf_counter

    from openai import AsyncOpenAI

    messages = [
        {'role': 'system', 'content': 'please response in markdown only.'},
        {'role': 'user', 'content': 'What is SSE? Please include a javascript code example.'},
    ]
    chunks = await AsyncOpenAI().chat.completions.create(
        model='gpt-4',
        messages=messages,
        stream=True,
    )

    last = None
    result_chunks = []
    async for chunk in chunks:
        now = perf_counter()
        if last is not None:
            t = now - last
        else:
            t = 0
        text = chunk.choices[0].delta.content
        # print(repr(text), t)
        if text is not None:
            result_chunks.append((t, text))
            log.debug(f'Queueing {text}')
            q.put_nowait(text)
        last = now

    q.put(None)  # All Done
    # log.debug('OpenAI Chat Queueing Done')
    # log.debug(result_chunks)
    text = ''.join(text for _, text in result_chunks)
    # log.debug(text)
    run_in_background(persist_openai_chat, messages, text)


def run_in_background(asyncfunc, *args):
    loop = asyncio.new_event_loop()
    loop.run_in_executor(None, async_to_sync(asyncfunc), *args)


@view_config(
    route_name='openaistream',
)
def openaistream(request):
    q = queue.Queue()
    # async_to_sync(run_openai)(q)
    # loop = asyncio.new_event_loop()
    # loop.run_in_executor(None, async_to_sync(run_openai), q)
    run_in_background(run_openai, q)

    def openai_resp_iter():
        sentinela = object()
        for resp in iter(q.get, sentinela):
            if resp is None:
                # break
                return StopIteration
            yield f'data: {resp}\n\n'.encode('utf-8')
            # sleep(0.1)

    return Response(
        app_iter=openai_resp_iter(),
        headerlist=[
            ('Content-Type', 'text/event-stream'),
            ('Cache-Control', 'no-cache')
        ]
    )


def build_app(debug=False):
    with Configurator() as config:
        # https: // docs.pylonsproject.org / projects / pyramid / en / latest / narr / security.html
        # checking-csrf-tokens-automatically
        # config.set_default_csrf_options(require_csrf=True)
        config.include('pyramid_jinja2')
        config.add_route('home', '/')
        config.add_route('procesar', '/procesar.php')
        config.add_route('doggo', '/doggo')
        config.add_route('dogstream', '/dogstream')
        config.add_route('openaistream', '/openaistream')
        config.scan()
        if debug:
            config.include('pyramid_debugtoolbar')
        app = config.make_wsgi_app()
    return app


app = build_app(debug=True)
asgi_app = WsgiToAsgi(build_app())

if __name__ == '__main__':
    # with Configurator() as config:
    #     config.include('pyramid_jinja2')
    #     config.add_route('home', '/')
    #     config.scan()
    #     config.include('pyramid_debugtoolbar')
    #     app = config.make_wsgi_app()
    # app = build_app()
    server = make_server('0.0.0.0', 6543, app)
    server.serve_forever()

    # import uvicorn
    #
    # uvicorn.run('app:asgi_app', host='0.0.0.0', port=6543, reload=True)
