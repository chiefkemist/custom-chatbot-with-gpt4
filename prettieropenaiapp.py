from __future__ import annotations

import asyncio
import logging
import sys
from typing import AsyncIterable

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastui import AnyComponent, FastUI, prebuilt_html
from fastui import components as c
from fastui.events import PageEvent
from starlette.responses import StreamingResponse

logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [%(module)s] %(message)s',
)
log = logging.getLogger(__name__)

app = FastAPI()


async def run_openai() -> AsyncIterable[str]:
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
    output = ''
    async for chunk in chunks:
        now = perf_counter()
        if last is not None:
            t = now - last
        else:
            t = 0
        text = chunk.choices[0].delta.content
        # print(repr(text), t)
        if text is not None:
            log.debug(f'Queueing {text}')
            output += text
            m = FastUI(root=[c.Markdown(text=output)])
            msg = f'data: {m.model_dump_json(by_alias=True, exclude_none=True)}\n\n'
            # await asyncio.sleep(0.4)
            yield msg
        last = now
    log.debug(output)
    # Hack to stop browser from hanging
    while True:
        yield ''
        await asyncio.sleep(0.4)


@app.get('/api/sse')
async def openapi_streaming_response() -> StreamingResponse:
    return StreamingResponse(
        run_openai(),
        media_type='text/event-stream',
    )


@app.get('/api/', response_model=FastUI, response_model_exclude_none=True)
def home() -> list[AnyComponent]:
    return [
        c.PageTitle(text='ChatApp | Home'),
        c.Page(
            components=[
                c.Heading(text='ChatApp'),
                c.Paragraph(text='Welcome to ChatApp!'),
                c.Div(
                    components=[
                        c.Heading(text='Server Load SSE', level=2),
                        c.Markdown(
                            text=(
                                '`ServerLoad` can also be used to load content from an SSE stream.\n\n'
                                "Here the response is the streamed output from OpenAI's GPT-4 chat model."
                            )
                        ),
                        c.Button(text='Load SSE content', on_click=PageEvent(name='server-load-sse')),
                        c.Div(
                            components=[
                                c.ServerLoad(
                                    path='/sse',
                                    sse=True,
                                    load_trigger=PageEvent(name='server-load-sse'),
                                    components=[c.Text(text='before')],
                                ),
                            ],
                            class_name='my-2 p-2 border rounded',
                        ),
                    ],
                    class_name='border-top mt-3 pt-1',
                ),
            ]
        )
    ]


@app.get('/{path:path}', status_code=404)
async def landing_page() -> HTMLResponse:
    return HTMLResponse(prebuilt_html(title='Pretty SSE App Demo'))


# @app.get('/path:path', status_code=404)
# async def not_found(path: str):
#     return FastUI(
#         c.h1('404 Not Found'),
#         c.p(f'Path: {path}'),
#     )

if __name__ == '__main__':
    import uvicorn

    uvicorn.run('prettieropenaiapp:app', host='0.0.0.0', port=6543, reload=True)
