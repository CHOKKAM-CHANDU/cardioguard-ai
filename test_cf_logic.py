import urllib.request, json

# HIGH RISK patient but NON-SMOKER (cigsPerDay=0)
payload = json.dumps({
    'age':65,'sex':1,'cigsPerDay':0,'BPMeds':1,'prevalentStroke':0,
    'prevalentHyp':1,'diabetes':0,'totChol':300,'BMI':32,'heartRate':85,
    'glucose':110,'pulse_pressure':80,'education':1
}).encode()

print('Testing /counterfactual with HIGH RISK NON-SMOKER...')
req = urllib.request.Request('http://localhost:8000/counterfactual', data=payload,
    headers={'Content-Type':'application/json'}, method='POST')

try:
    res = urllib.request.urlopen(req, timeout=60)
    d = json.loads(res.read())
    print('Response:', json.dumps(d, indent=2))
    if d.get('is_low_risk'):
        print('Patient is already low risk')
    elif 'error' in d:
        print('Server error:', d['error'])
    else:
        print(f"Found {len(d['pathways'])} pathways.")
except Exception as e:
    print(f"Request Error: {e}")
