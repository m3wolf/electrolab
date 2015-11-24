def display_progress(objs, operation='Status'):
    """
    Display the progress of the current operation via print statements.
    """
    ctr = 1
    total = len(objs)
    for obj in objs:
        status = '{operation}: {curr}/{total} ({percent:.0f}%)'.format(
            operation=operation, curr=ctr, total=total, percent=(ctr)/total*100
        )
        print(status, end='\r')
        ctr += 1
        yield obj
    print() # Newline to avoid writing over progress
