import requests
import os
import argparse

ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN')

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--upload_file', dest='upload_file', type = str)

    headers = {"Content-Type": "application/json"}
    params = {'access_token': ACCESS_TOKEN}

    r = requests.post('https://sandbox.zenodo.org/api/deposit/depositions',
                    params=params,
                    json={},
                    # Headers are not necessary here since "requests" automatically
                    # adds "Content-Type: application/json", because we're using
                    # the "json=" keyword argument
                    # headers=headers,
                    headers=headers)

    bucket_url = r.json()["links"]["bucket"]
    filename = args.upload_file.rsplit('/', 1)[1]
    print('uploading data now...')
    with open(args.upload_file, "rb") as fp:
        r = requests.put(
            "%s/%s" % (bucket_url, filename),
            data=fp,
            params=params,
        )
    print('uploading finished')

    data = {
         'metadata': {
             'title': 'MultiDCP data',
             'upload_type': 'poster',
             'description': 'MultiDCP data',
             'creators': [{'name': 'Qiao, Liu',
                           'affiliation': 'Hunter College, City University of New York'}]
         }
     }
    print(r.status_code)
