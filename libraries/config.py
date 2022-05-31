import yaml

data = dict(
    act1 = '021',
    act2 = '024',
    act3 ='026',
    act4 ='036',
    act5 ='043',
    act6 ='045',
    act7 ='046',
    act8 ='055',
    act9 ='059',
    act10 ='111',
    act11 ='113',
    act12 ='117',
    act13 ='124',
    act14 ='132',
    act15 ='136',
    act16 ='140',
    act17 ='143',
    act18 ='152',
    act19 ='153',
    act20 ='154',
    act21 ='209',
    act22 = '228',
    act23 = '236'
    
)

with open('data/times.yaml', 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)
