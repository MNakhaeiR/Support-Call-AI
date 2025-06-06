import asyncio
import websockets
import json

class WebSocketServer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.connected_clients = set()

    async def handler(self, websocket, path):
        self.connected_clients.add(websocket)
        try:
            async for message in websocket:
                await self.process_message(message)
        finally:
            self.connected_clients.remove(websocket)

    async def process_message(self, message):
        data = json.loads(message)
        # Process the incoming message and send a response if needed
        response = {"status": "received", "data": data}
        await self.send_response(response)

    async def send_response(self, response):
        if self.connected_clients:
            message = json.dumps(response)
            await asyncio.wait([client.send(message) for client in self.connected_clients])

    def start(self):
        start_server = websockets.serve(self.handler, self.host, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    server = WebSocketServer()
    server.start()