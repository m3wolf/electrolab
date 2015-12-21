from collections import namedtuple

xycoord = namedtuple('xycoord', ('x', 'y'))

def display_progress(objs, operation='Working'):
    """
    Display the progress of the current operation via print statements.
    """
    print("{operation}...".format(operation=operation), end='\r')
    ctr = 1
    total = len(objs)
    for obj in objs:
        status = '{operation}: {curr}/{total} ({percent:.0f}%)'.format(
            operation=operation, curr=ctr, total=total, percent=(ctr)/total*100
        )
        print(status, end='\r')
        ctr += 1
        yield obj
        # Newline to avoid writing over progress
    print('{operation}: {total}/{total} [done]'.format(operation=operation,
                                                       total=total))
