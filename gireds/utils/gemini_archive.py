import urllib
import json
import sys


def query_archive(query):
    # Construct the URL. We'll use the jsonsummary service
    url = 'https://archive.gemini.edu/jsonsummary/canonical/'
    
    # List the OBJECT files taken with GMOS-N on 2010-12-31
    url += query
    
    # Open the URL and fetch the JSON document text into a string
    u = urllib.urlopen(url)
    jsondoc = u.read()
    u.close()
    
    # Decode the JSON
    files = json.loads(jsondoc)
    
    # This is a list of dictionaries each containing info about a file
    total_data_size = 0
    print(
        '{:30s}{:12s}{:12s}{:12s}{:16s}{:8s}{:>10s}'.format(
            'Filename', 'Obs. Class', 'Obs. Type', 'Qa state',
            'Object Name', 'CWL (nm)', 'Disperser'))
    
    for f in files:
        if f['central_wavelength'] is None:
            f['central_wavelength'] = 0
        else:
            f['central_wavelength'] *= 1e+3

        fields = [
            f['name'], f['ra'], f['dec'],
            f['observation_class'], f['observation_type'],
            f['qa_state'],  f['object'], f['central_wavelength'],
            f['disperser']]
        
        total_data_size += f['data_size']
        print('{:30s}{:8.2f}{:8.2f}{:12s}{:12s}{:12s}{:16s}{:8.0f}{:>10s}'.
              format(*fields))
    
    print 'Total data size: %d' % total_data_size


if __name__ == '__main__':
    
    query_archive(sys.argv[1])
