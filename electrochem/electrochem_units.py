import units

# Prepare some units specific to electrochemistry
hour = units.unit('h')
mAh = units.named_unit('mAh', ['mA', 'h'], [])
uAh = units.scaled_unit('µAh', 'mAh', 10**-3)
units.named_unit('µA', ['uA'], [])
