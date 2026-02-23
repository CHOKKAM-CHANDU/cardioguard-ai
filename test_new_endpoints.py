import urllib.request, json

# MODERATE patient - age 50 with some risk factors
payload = json.dumps({
    'age':50,'sex':1,'cigsPerDay':15,'BPMeds':0,'prevalentStroke':0,
    'prevalentHyp':0,'diabetes':0,'totChol':220,'BMI':26,'heartRate':75,
    'glucose':85,'pulse_pressure':50,'education':2
}).encode()

print('Testing /simulate with MODERATE patient...')
req = urllib.request.Request('http://localhost:8000/simulate', data=payload,
    headers={'Content-Type':'application/json'}, method='POST')
d = json.loads(urllib.request.urlopen(req, timeout=60).read())
print('sim_img bytes:', len(d.get('sim_img', '')))
print('Year | Worst% | Best%')
print('-'*30)
for i, row in enumerate(d['data']):
    if i % 5 == 0:  # print every 5 years
        print(f"  yr={i:2d}  worst={row['worst']:5.1f}%  best={row['best']:5.1f}%")
