#!/usr/bin/env python3

import asyncio
import functools
import concurrent.futures
import logging
import sys
import queue
import uuid

from time import sleep

import httpx
import markdown
from asgiref.sync import async_to_sync
from asgiref.wsgi import WsgiToAsgi
from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.httpexceptions import HTTPFound
from pyramid.response import Response
# from pyramid.httpexceptions import HTTPFound
from pyramid.view import view_config
# from pyramid.renderers import render
from pyramid.csrf import get_csrf_token, CookieCSRFStoragePolicy
from jinja2 import Environment, FileSystemLoader, select_autoescape

logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [%(module)s] %(message)s',
)
log = logging.getLogger(__name__)


# log.addHandler(logging.StreamHandler(sys.stdout)

@view_config(
    route_name='index',
    renderer='templates/index.jinja2',
)
def index(request):
    req_id = str(uuid.uuid4())
    messages = [
        {'sender': 'ai', 'content': 'Hi, how can I help you today?'},
        {'sender': 'user', 'content': 'faslskadalksjioqjeqlkj'},
        {'sender': 'ai',
         'content': 'Sorry, I couldn\'t find any information in the documentation about that. Expect answer to be less accurateI could not find the answer to this in the verified sources.'},
    ]
    # TODO: Add div to last AI block to trigger rendering of SSE response
    return dict(messages=messages, req_id=req_id)


reqs = {}


@view_config(
    route_name='openai',
    request_method='POST',
    renderer='templates/index.jinja2',
)
def openai(request):
    # TODO: Persist the user prompt
    # TODO: Return an htmx div for sse that will submit the user prompt by its data-id
    # TODO: Stream down the response
    # TODO: Persist the full response and terminate the stream cleanly

    # home_url = request.route_url('index')
    # return HTTPFound(location=home_url, code=303)

    req_id = str(uuid.uuid4())

    user_prompt = request.POST.get('user_prompt')
    log.debug(f'User prompt: {user_prompt}')

    req_id = request.matchdict['req_id']
    reqs[req_id] = user_prompt

    messages = [
        {'sender': 'user', 'content': user_prompt},
        {'sender': 'ai', 'content': ''},
        # {'sender': 'ai', 'content': 'Thinking...'},
    ]

    sse_config = dict(
        listener='openai',
        path=f'/openaistream/{req_id}',
        # path='/openaistream',
        topics=[
            'Response',
            'ResponseNoMass',
            'Terminate',
        ]
    )

    return dict(messages=messages, req_id=req_id, sse_config=sse_config)


def render_sse_html_chunk(event, chunk, attrs=None):
    if attrs is None:
        attrs = {}
    tmpl = Environment(
        loader=FileSystemLoader('templates/partials'),
        autoescape=select_autoescape(['html'])
    ).select_template(['streaming_chunk.jinja2'])
    html_chunk = tmpl.render(**dict(event=event, chunk=chunk, attrs=attrs))
    return html_chunk


def run_in_background(asyncfunc, *args):
    loop = asyncio.new_event_loop()
    loop.run_in_executor(None, async_to_sync(asyncfunc), *args)


# def run_in_background(asyncfunc, *args):
#     with concurrent.futures.ThreadPoolExecutor() as pool:
#         loop = asyncio.new_event_loop()
#         try:
#             loop.run_in_executor(pool, async_to_sync(asyncfunc), *args)
#         finally:
#             loop.close()


# async def run_openai(q: queue.Queue):
async def run_openai(req_id: str, q: queue.Queue):
    from time import perf_counter

    from openai import AsyncOpenAI

    # user_prompt = 'What do you do?'
    # user_prompt = 'What up though?'
    user_prompt = reqs[req_id]
    log.debug(f'User prompt: {user_prompt}')

    messages = [
        {'role': 'system', 'content': 'please response in markdown only.'},
        {'role': 'user', 'content': user_prompt},
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
            # mdText = markdown.markdown(result_concat)
            # mdText = markdown.markdown(result_concat)
            # log.debug(f'Queueing {mdText}')
            # q.put_nowait(mdText)
            q.put_nowait(result_concat)
            # await asyncio.sleep(0.4)
        # else:
        #     log.debug('No text adding space')
        #     # await asyncio.sleep(0.2)
        #     q.put_nowait(' ')
        last = now

    # await asyncio.sleep(0.8)
    # q.put(None)  # All Done
    q.put(None)  # All Done
    q.join()
    # log.debug('OpenAI Chat Queueing Done')
    # log.debug(result_chunks)
    text = ''.join(text for _, text in result_chunks)


@view_config(
    route_name='openaistream',
)
def openaistream(request):
    req_id = request.matchdict['req_id']
    log.info(f"Request ID: {req_id}")
    # log.info(f"Request ID: {request.matchdict['req_id']}")

    q = queue.Queue()
    # run_in_background(run_openai, q)
    run_in_background(run_openai, req_id, q)
    # resp_gen = run_in_background(
    #     functools.partial(run_openai, req_id, q)
    # )

    # run_in_background(run_openai, [req_id, q])

    def openai_iter():
        response_parts = []
        sentinela = object()
        for resp in iter(q.get, sentinela):
            # for i in range(80):
            # sleep(0.2)
            # plain = f'Chunk {i}'
            # response_parts.append(plain)
            if resp is None:
                chunk = render_sse_html_chunk(
                    'Terminate',
                    '',
                    {
                        'id': 'openai',
                        'hx-swap-oob': 'true',
                    },
                )
                yield f'{chunk}\n\n'.encode('utf-8')
                return StopIteration
            # log.debug(f'Queueing {resp}')
            # response_parts.append(resp)
            # curr_resp = ''.join(response_parts)
            chunk = render_sse_html_chunk(
                'Response',
                # curr_resp,
                resp,
                {
                    'id': 'Response',
                    # 'hx-swap-oob': 'beforeend',
                    'hx-swap-oob': 'true',
                },
            )
            yield f'{chunk}\n\n'.encode('utf-8')
        chunk = render_sse_html_chunk(
            'Terminate',
            '',
            {
                'id': 'openai',
                'hx-swap-oob': 'true',
            },
        )
        yield f'{chunk}\n\n'.encode('utf-8')

    return Response(
        app_iter=openai_iter(),
        headerlist=[
            ('Content-Type', 'text/event-stream'),
            ('Cache-Control', 'no-cache')
        ]
    )


def build_app(debug=True):
    with Configurator() as config:
        # https: // docs.pylonsproject.org / projects / pyramid / en / latest / narr / security.html
        # checking-csrf-tokens-automatically
        config.set_default_csrf_options(require_csrf=True)
        config.set_csrf_storage_policy(CookieCSRFStoragePolicy())
        config.include('pyramid_jinja2')
        config.add_route('index', '/')
        config.add_route('openai', '/openai/{req_id}')
        config.add_route('openaistream', '/openaistream/{req_id}')
        # config.add_route('openaistream', '/openaistream')
        config.scan()
        if debug:
            config.include('pyramid_debugtoolbar')
        app = config.make_wsgi_app()
    return app


asgi_app = WsgiToAsgi(build_app())
asgi_app_dev = WsgiToAsgi(build_app(debug=True))

if __name__ == '__main__':
    # app = build_app(debug=True)
    # server = make_server('0.0.0.0', 6543, app)
    # server.serve_forever()

    import uvicorn

    uvicorn.run('pyramid_chat:asgi_app', host='0.0.0.0', port=6543, reload=True)
    # uvicorn.run('pyramid_chat:asgi_app_dev', host='0.0.0.0', port=6543, workers=4)
