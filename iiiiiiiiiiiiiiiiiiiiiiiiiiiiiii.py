988cb26337c4   nginx:alpine                     "/docker-entrypoint.…"   
c3b2530bdcc7   unbreakable-eye-fastapi          "uvicorn src.web.mai…"   
fb0addc73db7   pgvector/pgvector:pg16           "docker-entrypoint.s…"   


I need you explain these a little and I think this is a good topic for the techincal article:
    "# Three-phase pipeline restart engine"


explain this abit too:
    async def lifespan(app: FastAPI):

