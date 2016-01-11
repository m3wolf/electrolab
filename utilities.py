from collections import namedtuple
import sys

from tqdm import tqdm_gui, tqdm

xycoord = namedtuple('xycoord', ('x', 'y'))
Pixel = namedtuple('Pixel', ('vertical', 'horizontal'))

def prog(*args, **kwargs):
    """Progress meter. Wraps around tqdm with some custom defaults."""
    kwargs['file'] = kwargs.get('file', sys.stdout)
    kwargs['leave'] = kwargs.get('leave', True)
    return tqdm(*args, **kwargs)

# def prog(objs, operation='Working'):
#     """
#     Display the progress of the current operation via print statements.
#     """
#     print("{operation}...".format(operation=operation), end='\r')
#     ctr = 1
#     total = len(objs)
#     for obj in objs:
#         status = '{operation}: {bar} {curr}/{total} ({percent:.0f}%)'.format(
#             operation=operation, bar=progress_bar(ctr, total),
#             curr=ctr, total=total, percent=(ctr)/total*100
#         )
#         print(status, end='\r')
#         ctr += 1
#         yield obj
#     # print('{operation}: {bar} {total}/{total} [done]'.format(
#     #     operation=operation, bar=progress_bar(total, total), total=total))
