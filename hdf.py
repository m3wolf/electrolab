class HDFAttribute():
    """A descriptor that returns an HDF5 attribute if possible or else an
    in-memory value. An optional `wrapper` argument will wrap the data
    in this function before returning it. Assumes the object to which
    it is attached has an attribute/property `hdf_node` that returns a
    node object.
    """
    def __init__(self, attribute_name, default=None, wrapper=None):
        self.attribute_name = attribute_name
        self.default_value = default
        self.wrapper = wrapper

    def __get__(self, obj, owner):
        attrs = getattr(obj.hdf_node(), 'attrs', obj._attrs)
        value = attrs.get(self.attribute_name, self.default_value)
        if self.wrapper:
            value = self.wrapper(value)
        return value

    def __set__(self, obj, value):
        attrs = getattr(obj.hdf_node(), 'attrs', obj._attrs)
        attrs[self.attribute_name] = value
