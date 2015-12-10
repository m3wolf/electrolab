class HDFAttribute():
    """A descriptor that returns an HDF5 attribute if possible or else an
    in-memory value. An optional `wrapper` argument will wrap the data
    in this function before returning it. Assumes the object to which
    it is attached has an attribute/property `hdf_node` that returns a
    node object.

    Arguments
    ---------
    attribute_name (string): name to be assigned to the HDF5 attribute
        storing this datum

    default (any): Default value to return if no value is stored

    wrapper (callable): Returned values get sent through this callable first

    group_func (string): Name of object attribute to call in order to
        retrieve the hdf5 group that this attr should be saved
        to. Default 'hdf_node'.
    """
    def __init__(self, attribute_name, default=None, wrapper=None,
                 group_func='hdf_node'):
        self.attribute_name = attribute_name
        self.default_value = default
        self.wrapper = wrapper
        self.group_func = group_func

    def _attrs(self, obj):
        # Retrieve the HDF5 group
        getter = getattr(obj, self.group_func)
        # Retrieve the attrs dictionary
        attrs = getattr(getter(), 'attrs', obj._attrs)
        return attrs

    def __get__(self, obj, owner):
        attrs = self._attrs(obj)
        value = attrs.get(self.attribute_name, self.default_value)
        if self.wrapper:
            value = self.wrapper(value)
        return value

    def __set__(self, obj, value):
        attrs = self._attrs(obj)
        attrs[self.attribute_name] = value
