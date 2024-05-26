#!/usr/bin/env python3

import asyncio
import functools
import concurrent.futures
import logging
import re
import sys
import queue
import typing
import uuid
from io import StringIO

from time import sleep

import httpx
import markdown
# from pygments.formatters import HtmlFormatter
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters.html import HtmlFormatter

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [%(module)s] %(message)s',
)
log = logging.getLogger(__name__)

# log.addHandler(logging.StreamHandler(sys.stdout)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

def app_context(request: Request) -> typing.Dict[str, typing.Any]:
    return {'app': request.app}

templates = Jinja2Templates(
    directory="fasttemplates",
    context_processors=[app_context],
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    req_id = str(uuid.uuid4())
    messages = [
        {'sender': 'ai', 'content': 'Hi, how can I help you today?'},
        {'sender': 'user', 'content': 'faslskadalksjioqjeqlkj'},
        {'sender': 'ai',
         'content': 'Sorry, I couldn\'t find any information in the documentation about that. Expect answer to be less accurateI could not find the answer to this in the verified sources.'},
    ]
    # TODO: Add div to last AI block to trigger rendering of SSE response
    return templates.TemplateResponse(
        request=request, name="index.jinja2",
        context=dict(messages=messages, req_id=req_id)
    )


reqs = {}


@app.post("/openai/{req_id}", response_class=HTMLResponse)
async def openai(request: Request, req_id: str, user_prompt: typing.Annotated[str, Form()]):
    # TODO: Persist the user prompt
    # TODO: Return an htmx div for sse that will submit the user prompt by its data-id
    # TODO: Stream down the response
    # TODO: Persist the full response and terminate the stream cleanly
    # TODO: Add a button to terminate the stream
    # TODO: Add image support
    # TODO: Add commands to the prompt: /chat, /image, /help etc.
    # TODO: Setup OpenAI key permisisions
    # TODO: Setup Semantic Router

    # home_url = request.route_url('index')
    # return HTTPFound(location=home_url, code=303)

    log.debug(f'User prompt: {user_prompt}')

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

    new_req_id = str(uuid.uuid4())

    return templates.TemplateResponse(
        request=request, name="index.jinja2",
        context=dict(
            messages=messages, req_id=new_req_id, sse_config=sse_config
        )
    )


def render_sse_html_chunk(event, chunk, attrs=None):
    if attrs is None:
        attrs = {}
    tmpl = Environment(
        loader=FileSystemLoader('templates/partials'),
        autoescape=select_autoescape(['html'])
    ).select_template(['streaming_chunk.jinja2'])
    html_chunk = tmpl.render(**dict(event=event, chunk=chunk, attrs=attrs))
    return html_chunk


def markdown_to_html_with_highlighting(source_markdown):
    # Configure Markdown to use the 'fenced_code' extension with Pygments
    md = markdown.Markdown(extensions=['fenced_code', 'codehilite'])

    # Convert Markdown to HTML
    html = md.convert(source_markdown)

    # Generate CSS for syntax highlighting
    css = HtmlFormatter().get_style_defs('.codehilite')

    # return css + html
    return f"<style>{css.replace('\n', '   ')}</style>{html.replace('\n', '<br />')}"


def markdown_to_html_with_inline_highlighting(source_markdown):
    # Create an instance of HtmlFormatter with inline styles
    formatter = HtmlFormatter(style='default', cssclass='', noclasses=True)

    # Custom inline code highlighter
    def inline_highlight(match):
        language = match.group('lang')
        code = match.group('code')
        lexer = get_lexer_by_name(language, stripall=True)
        # highlighted_code = highlight(code, lexer, formatter)
        highlighted_code = highlight(code.replace('<br />', '\n'), lexer, formatter)
        return highlighted_code.replace('\n', '<br />')
        # return highlighted_code.replace('&lt;br /&gt;', '<br />')
        return highlighted_code

    # Replace fenced code blocks with highlighted code
    highlighted_markdown = re.sub(
        r'```(?P<lang>\w+)\s*(?P<code>.*?)```',
        inline_highlight,
        source_markdown,
        flags=re.DOTALL
    )

    # Convert Markdown to HTML
    html = markdown.markdown(highlighted_markdown)

    return html

async def run_openai(req_id: str):
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
    # result_concat = ''
    result_concat = StringIO()
    code_start = False
    tick_count = 0
    async for chunk in chunks:
        now = perf_counter()
        if last is not None:
            t = now - last
        else:
            t = 0
        text = chunk.choices[0].delta.content
        # print(repr(text), t)
        # log.debug(f'Chunk: {text}')
        if text is not None:
            result_chunks.append((t, text))
            # log.debug(f'Queueing {text}')
            # q.put_nowait(text)
            # result_concat += text
            # result_concat.write(text)
            # mdText = markdown.markdown(result_concat)
            # mdText = markdown.markdown(result_concat)
            # log.debug(f'Queueing {mdText}')
            # q.put_nowait(mdText)
            # yield result_concat
            # mdText = markdown.markdown(result_concat)
            # if text == '`':
            #     tick_count += 1
            # if text == '`' and tick_count == 3:
            #     code_start = not code_start
            #     log.debug(f'Code start: {code_start}')
            #     tick_count = 0
            # if code_start is True:
            #     result_concat.write(f"{text}".replace('\n', "##"))
            # else:
            #     result_concat.write(f"{text}".replace('\n', "<br />"))
            # mdText = markdown.markdown(result_concat.getvalue())
            result_concat.write(f"{text}")
            # markdown.markdown(result_concat.getvalue().replace('\n', "<br />"))
            # mdText = replace_br_with_newline(
            #     markdown.markdown(result_concat.getvalue().replace('\n', "<br />"))
            # )
            # mdText = markdown.markdown(
            #     result_concat.getvalue()
            #     .replace('\n', "<br />")
            # ).replace('&lt;br /&gt;', '<br />')
            # mdText = markdown_to_html_with_highlighting(
            #     result_concat.getvalue()
            #     .replace('\n', "<br />")
            # ).replace('&lt;br /&gt;', '<br />')
            mdText = markdown_to_html_with_inline_highlighting(
                result_concat.getvalue().replace('\n', "<br />")
            )
            yield mdText
            # await asyncio.sleep(0.4)
        else:
            log.debug('No text adding space')
            # await asyncio.sleep(0.2)
            # q.put_nowait(' ')
            # result_concat += ' '
            # result_concat += '\n'
            # result_concat.write(' ')
            # # mdText = markdown.markdown(result_concat)
            # mdText = markdown.markdown(result_concat.getvalue())
            # yield mdText
        last = now

    # log.debug(f'Final result: {result_concat.getvalue()}')
    # yield markdown.markdown(result_concat.getvalue().replace('\n', '<br />'))

    # await asyncio.sleep(0.8)
    # q.put(None)  # All Done
    yield None  # All Done
    # log.debug('OpenAI Chat Queueing Done')
    # log.debug(result_chunks)
    text = ''.join(text for _, text in result_chunks)


@app.get("/openaistream/{req_id}", response_class=StreamingResponse)
async def openaistream(request: Request, req_id: str):
    log.info(f"Request ID: {req_id}")
    # log.info(f"Request ID: {request.matchdict['req_id']}")

    async def openai_iter():
        response_parts = []
        async for resp in run_openai(req_id):
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
                # raise StopAsyncIteration
                break
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

    return StreamingResponse(
        openai_iter(),
        media_type='text/event-stream',
    )

if __name__ == '__main__':
    import uvicorn

    uvicorn.run('async_chat:app', host='0.0.0.0', port=6543, reload=True)
    # uvicorn.run('async_chat:app', host='0.0.0.0', port=6543, workers=4)
