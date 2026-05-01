import time
import threading
import queue

def data_generator():
    for i in range(10):
        time.sleep(0.1) # Simulate I/O
        yield i

def process(item):
    time.sleep(0.1) # Simulate tokenization

def synchronous():
    t0 = time.time()
    for item in data_generator():
        process(item)
    print(f"Sync: {time.time()-t0:.2f}s")

def threaded():
    t0 = time.time()
    q = queue.Queue(maxsize=3)
    def producer():
        for item in data_generator():
            q.put(item)
        q.put(None)
    
    t = threading.Thread(target=producer)
    t.start()
    
    while True:
        item = q.get()
        if item is None: break
        process(item)
        
    t.join()
    print(f"Threaded: {time.time()-t0:.2f}s")

synchronous()
threaded()
