#!/usr/bin/env python3

import asyncio
import logging
import sys
import typing

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

# from pygments.formatters import HtmlFormatter

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
    directory="templates",
    context_processors=[app_context],
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="openapi.jinja2",
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


async def gen_dog_breeds():
    async with httpx.AsyncClient() as client:
        breeds = (await client.get('https://dog.ceo/api/breeds/list/all')).json()
        for breed in breeds['message'].keys():
            log.info(f"Yielding {breed}")
            yield breed


@app.get("/dogstream", response_class=StreamingResponse)
async def dogstream(request: Request):
    async def dogbreeds_iter():
        async for breed in gen_dog_breeds():
            await asyncio.sleep(0.2)
            breed_status_chunk = render_sse_html_chunk(
                'DogBreedNoMass',
                'More doggo senior :-)',
                {
                    'id': 'DogBreedNoMass',
                    # 'hx-swap-oob': 'beforeend',
                    'hx-swap-oob': 'true',
                },
            )
            yield f'{breed_status_chunk}\n\n'.encode('utf-8')
            await asyncio.sleep(0.2)
            chunk = render_sse_html_chunk(
                'DogBreed',
                breed,
                {
                    'id': 'DogBreed',
                    # 'hx-swap-oob': 'beforeend',
                    'hx-swap-oob': 'true',
                },
            )
            yield f'{chunk}\n\n'.encode('utf-8')
        breed_status_chunk = render_sse_html_chunk(
            'DogBreedNoMass',
            'No more doggo senior :-(',
            {
                'id': 'DogBreedNoMass',
                'hx-swap-oob': 'true',
            },
        )
        yield f'{breed_status_chunk}\n\n'.encode('utf-8')
        # chunk = render_sse_html_chunk(
        #     'Terminate',
        #     '',
        #     {
        #         'id': 'doggo-sse-listener',
        #         'hx-swap-oob': 'true',
        #     },
        # )
        # yield f'{chunk}\n\n'.encode('utf-8')

    return StreamingResponse(
        dogbreeds_iter(),
        media_type='text/event-stream',
    )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('async_doggo:app', host='0.0.0.0', port=6543, reload=True)
    # uvicorn.run('async_doggo:app', host='0.0.0.0', port=6543, workers=4)
