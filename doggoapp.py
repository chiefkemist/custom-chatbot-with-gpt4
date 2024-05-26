#!/usr/bin/env python3

import asyncio
import logging
import sys
import queue

import httpx
from asgiref.sync import async_to_sync
from asgiref.wsgi import WsgiToAsgi
from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.httpexceptions import HTTPFound
from pyramid.view import view_config
# from pyramid.renderers import render
from jinja2 import Environment, FileSystemLoader, select_autoescape

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
        breed_concat = ''
        for breed in breeds_list:
            log.debug(f'Queueing {breed}')
            q.put_nowait(breed)
            # log.debug(f'Queueing {breed_concat}')
            # breed_concat += breed + '##'  # Append to breeds concatenated list to avoid replacing the previous breed on the client
            # q.put_nowait(breed_concat)
            # sleep(1)
            await asyncio.sleep(0.2)
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
        tmpl = Environment(loader=FileSystemLoader('templates/partials'), autoescape=select_autoescape(['html'])).select_template(['streaming_chunk.jinja2'])
        sentinela = object()
        for breed in iter(q.get, sentinela):
            if breed is None:
                # yield f'event: DogBreedNoMass\ndata: No more doggo senior :-(\n\n'.encode('utf-8')
                chunk = tmpl.render(
                    # 'templates/partials/streaming_chunk.jinja2',
                    **dict(
                        event='DogBreed',
                        attrs={
                            'id': 'DogBreedNoMass',
                            'hx-swap-oob': 'true',
                        },
                        chunk='No more doggo senior :-(',
                    )
                )
                print(f'{chunk}\n\n'.encode('utf-8'))
                yield f'{chunk}\n\n'.encode('utf-8')
                chunk = tmpl.render(
                    # 'templates/partials/streaming_chunk.jinja2',
                    **dict(
                        event='Terminate',
                        attrs={
                            'id': 'doggo-sse-listener',
                            'hx-swap-oob': 'true',
                        },
                    )
                )
                print(f'{chunk}\n\n'.encode('utf-8'))
                yield f'{chunk}\n\n'.encode('utf-8')
                # break
                return StopIteration
            else:
                # yield f'event: DogBreedNoMass\ndata: More doggo senior :-)\n\n'.encode('utf-8')
                chunk = tmpl.render(
                    # 'templates/partials/streaming_chunk.jinja2',
                    **dict(
                        event='DogBreed',
                        attrs={
                            'id': 'DogBreedNoMass',
                            'hx-swap-oob': 'true',
                        },
                        chunk='More doggo senior :-)',
                    )
                )
                print(f'{chunk}\n\n'.encode('utf-8'))
                yield f'{chunk}\n\n'.encode('utf-8')
            # yield f'event: DogBreed\ndata: {breed}\n\n'.encode('utf-8')
            # yield f'event: DogBreed\ndata: {breed}\n\n'.encode('utf-8')
            chunk = tmpl.render(
                # 'templates/partials/streaming_chunk.jinja2',
                **dict(
                    event='DogBreed',
                    attrs={
                        'id': 'DogBreed',
                        'hx-swap-oob': 'true',
                    },
                    chunk=breed,
                )
            )
            print(f'{chunk}\n\n'.encode('utf-8'))
            yield f'{chunk}\n\n'.encode('utf-8')
            # sleep(1)

    breeds_gen = dog_breeds_iter()

    return Response(
        app_iter=breeds_gen,
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


def run_in_background(asyncfunc, *args):
    loop = asyncio.new_event_loop()
    loop.run_in_executor(None, async_to_sync(asyncfunc), *args)


def build_app(debug=False):
    with Configurator() as config:
        # https: // docs.pylonsproject.org / projects / pyramid / en / latest / narr / security.html
        # checking-csrf-tokens-automatically
        # config.set_default_csrf_options(require_csrf=True)
        config.include('pyramid_jinja2')
        config.add_route('doggo', '/')
        config.add_route('dogstream', '/dogstream')
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
