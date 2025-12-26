
import aiohttp
import asyncio
import base64
import json
import sys

async def verify():
    url = "ws://localhost:8000/ws/telemetry"
    print(f"Connecting to {url}...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                print("Connected. Waiting for telemetry...")
                
                # Wait for up to 10 messages or timeout
                for _ in range(20):
                    try:
                        msg = await ws.receive()
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data.get("type") == "race_replay" and data.get("format") == "binary_base64":
                                b64 = data.get("payload")
                                b_data = base64.b64decode(b64)
                                size = len(b_data)
                                print(f"Received binary payload. Size: {size} bytes")
                                
                                # Check Divisibility
                                if size > 0:
                                    # Header 16 bytes
                                    # Frame body = Size - (NumFrames * FrameHeader?) 
                                    # No, binary format is: [Header 16 + Pods...][Header 16 + Pods...] ??
                                    # Let's check telemetry.py logic.
                                    # flush_chunk: joins frames. 
                                    # pack_frame: Header(16) + Pods(4*40=160) = 176.
                                    # So total size should be divisible by 176.
                                    
                                    if size % 176 == 0:
                                        count = size // 176
                                        print(f"SUCCESS: Payload size {size} is divisible by 176. (Frames: {count})")
                                        print("Telemetry format verified as UPDATED (Shield/Boost included).")
                                        return True
                                    elif size % 144 == 0:
                                        count = size // 144
                                        print(f"FAILURE: Payload size {size} is divisible by 144. (Frames: {count})")
                                        print("Telemetry format is OLD (No Shield/Boost). Server restart required.")
                                        return False
                                    else:
                                        print(f"WARNING: Payload size {size} is not divisible by 176 or 144.")
                                        return False
                                        
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            print("Connection closed by server.")
                            break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            print("WS Error")
                            break
                            
                    except Exception as e:
                        print(f"Error receiving: {e}")
                        break
                        
        print("Timeout or no binary frames received.")
        return False
        
    except Exception as e:
        print(f"Connection Failed: {e}")
        return False

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    success = loop.run_until_complete(verify())
    sys.exit(0 if success else 1)
