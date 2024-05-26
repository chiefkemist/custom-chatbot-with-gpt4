#!/usr/bin/env python3

import asyncio
import logging
import sys
import queue
from time import sleep

import markdown

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


async def persist_openai_chat(query, response):
    log.debug(f'Persisting {query} => {response}')


async def run_openai(q: queue.Queue):
    from time import perf_counter

    from openai import AsyncOpenAI

    messages = [
        {'role': 'system', 'content': 'please response in markdown only.'},
        # {'role': 'user', 'content': 'What is SSE? Please include a javascript code example.'},
        {'role': 'user', 'content': 'Give me a code example of dynamic programming in Python. Answer with just markdown formatted code, no need for comments or anything. Just code! Only code!'},
    ]
    chunks = await AsyncOpenAI().chat.completions.create(
        model='gpt-4',
        messages=messages,
        stream=True,
    )

    last = None
    result_chunks = []
    result_concat = ''
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
            # log.debug(f'Queueing {text}')
            # q.put_nowait(text)
            result_concat += text
            mdText = markdown.markdown(result_concat)
            log.debug(f'Queueing {mdText}')
            q.put_nowait(mdText)
            # await asyncio.sleep(0.4)
        else:
            log.debug('No text adding space')
            # await asyncio.sleep(0.2)
            q.put_nowait(' ')
        last = now

    # await asyncio.sleep(0.8)
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
                yield f'event: ResponseNoMass\ndata: Done answering senior :-)\n\n'.encode('utf-8')
                # break
                return StopIteration
            else:
                yield f'event: ResponseNoMass\ndata: Thinking senior :-|\n\n'.encode('utf-8')
            yield f'event: Response\ndata: {resp}\n\n'.encode('utf-8')

    return Response(
        app_iter=openai_resp_iter(),
        headerlist=[
            ('Content-Type', 'text/event-stream'),
            ('Cache-Control', 'no-cache')
        ]
    )


@view_config(
    route_name='openai',
    renderer='templates/openai.jinja2'
)
def openai(request):
    return {}


def build_app(debug=False):
    with Configurator() as config:
        # https: // docs.pylonsproject.org / projects / pyramid / en / latest / narr / security.html
        # checking-csrf-tokens-automatically
        # config.set_default_csrf_options(require_csrf=True)
        config.include('pyramid_jinja2')
        config.add_route('openai', '/')
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
