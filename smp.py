from queue import Empty
import multiprocessing as mp
from time import time

from tqdm import format_meter

class Consumer(mp.Process):
    def __init__(self, target, task_queue, result_queue, **kwargs):
        ret = super().__init__(target=target, **kwargs)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.target = target
        return ret

    def run(self):
        """Retrieve and process a frame from the queue or exit if poison pill
        None is passed."""
        while True:
            payload = self.task_queue.get()
            if payload is None:
                # Poison pill, so exit
                self.task_queue.task_done()
                break
            result = self.target(payload)
            self.result_queue.put(result)
            self.task_queue.task_done()


class Queue():
    def __init__(self, worker, totalsize, result_callback=None, description="Processing data"):
        self.num_consumers = mp.cpu_count() * 2
        self.result_queue = mp.Queue(maxsize=totalsize)
        self.result_callback = result_callback
        self.totalsize = totalsize
        self.results_left = totalsize
        self.description = description
        self.task_queue = mp.JoinableQueue(maxsize=self.num_consumers)
        self.start_time = time()
        # Create all the worker processes
        self.consumers = [Consumer(target=worker,
                                   task_queue=self.task_queue,
                                   result_queue=self.result_queue)
                          for i in range(self.num_consumers)]
        for consumer in self.consumers:
            consumer.start()

    def put(self, obj, *args, **kwargs):
        # Check for results to take out of the queue
        try:
            result = self.result_queue.get(block=False)
        except Empty:
            pass
        else:
            self.process_result(result)
        return self.task_queue.put(obj, *args, **kwargs)

    def process_result(self, result):
        ret = self.result_callback(result)
        self.results_left -= 1
        curr = self.totalsize - self.results_left
        # Prepare a status bar
        status = format_meter(n=curr,
                              total=self.totalsize,
                              elapsed=time()-self.start_time,
                              prefix=self.description + ": ")
        # status = '{description}: {bar} {curr}/{total} ({percent:.0f}%)'.format(
        #     description=self.description,
        #     bar=progress_bar(current=curr, total=self.totalsize),
        #     curr=curr,
        #     total=self.totalsize,
        #     percent=(1 - (self.results_left/self.totalsize)) * 100
        # )
        print(status, end='\r')
        return ret

    def join(self):
        # Send poison pill to all consumers
        for i in range(self.num_consumers):
            self.task_queue.put(None)
        ret = self.task_queue.join()
        # Finish emptying the results queue
        while self.results_left > 0:
            result = self.result_queue.get()
            self.process_result(result)
        # Display "finished" message
        # print('{description}: {bar} {total}/{total} [done]'.format(
        #     description=self.description,
        #     bar=progress_bar(self.totalsize, self.totalsize),
        #     total=self.totalsize))
        print()
        return ret
