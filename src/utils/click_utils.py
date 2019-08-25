def conditionally_decorate(decorator, condition):
    '''return decorator that applies a passed decorator if some condition is True

    See https://stackoverflow.com/questions/3773555/python3-decorating-conditionally
    '''
    def resdec(f):
        if not condition:
            return f
        return decorator(f)
    return resdec
