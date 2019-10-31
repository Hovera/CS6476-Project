import requests
import base64
import json

class PlateReader:
    def process_image(self, img_path):
        raise NotImplementedError


class OpenALPRReader(PlateReader):
    def __init__(self, skey):
        self.skey = skey

    def process_image(self, img_path, outpath=None):
        with open(img_path, 'rb') as f:
            img_data = base64.b64encode(f.read())

        # Query OpenALPR
        url = ('https://api.openalpr.com/v2/recognize_bytes?'
               'recognize_vehicle=1&country=us&secret_key=%s' % (self.skey))
        r = requests.post(url, data=img_data)

        # Write json dump to file
        if outpath:
            dump = json.dumps(r.json(), indent=2)
            with open(outpath, 'w') as f:
                f.write(dump)

        return self._process_json(r.json())

    def _process_json(self, dump):
        if not dump['results']:
            return None, 0.

        plate = dump['results'][0]['plate']
        confidence = dump['results'][0]['confidence']
        
        return plate, confidence
