from argparse import ArgumentParser
import json
from datetime import datetime
import boto3


def make_hit_question(hit_set_id, experiment_url):
    url_for_hit = '{}?hitSetId={}'.format(experiment_url, hit_set_id)
    frame_height = '1150'
    xml = '<ExternalQuestion ' \
          'xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">' \
          '<ExternalURL>' + url_for_hit + '</ExternalURL> ' \
          '<FrameHeight>' + frame_height + '</FrameHeight> ' \
          '</ExternalQuestion>'
    return xml


def setup_mturk_connection(credentials, production=False):
    environments = {
      'production': {
        'endpoint': 'https://mturk-requester.us-east-1.amazonaws.com',
        'preview': 'https://www.mturk.com/mturk/preview'
      },
      'sandbox': {
        'endpoint': 'https://mturk-requester-sandbox.us-east-1.amazonaws.com',
        'preview': 'https://workersandbox.mturk.com/mturk/preview'
      },
    }

    if production:
        mturk_environment = environments['production']
    else:
        mturk_environment = environments['sandbox']

    mturk = boto3.client('mturk',
        aws_access_key_id = credentials['aws_access_key_id'],
        aws_secret_access_key = credentials['aws_secret_access_key'],
        region_name='us-east-1',
        endpoint_url=mturk_environment['endpoint'])

    return mturk


if __name__ == '__main__':
    parser = ArgumentParser(description='Automatically post hits')
    parser.add_argument('--num_hits', required=True, type=int, help='number of hits to post')
    parser.add_argument('--production', action='store_true', help='release hits to production environment')
    args = parser.parse_args()

    with open('credentials.json', 'r') as f:
        credentials = json.loads(f.read())
    mturk = setup_mturk_connection(credentials, production=args.production)

    with open('hit_params.json', 'r') as f:
        hit_params = json.loads(f.read())
    hit_type_id = mturk.create_hit_type(**hit_params['hit_type'])['HITTypeId']

    if args.production:
        experiment_url = 'https://scene-representation-gqn.s3.amazonaws.com/behavioural/index.html'
    else:
        experiment_url = 'https://scene-representation-gqn.s3.amazonaws.com/behavioural/index_sandbox.html'

    hit_ids = []
    for hit_set_id in range(args.num_hits):
        question = make_hit_question(hit_set_id, experiment_url)
        hit = mturk.create_hit_with_hit_type(Question=question, HITTypeId=hit_type_id,
                                             **hit_params['create_hit'])['HIT']
        hit_ids.append(hit['HITId'])

    with open('created_hits/' + datetime.utcnow().strftime("%d:%m:%Y_%H:%M:%S") + '.txt', 'w') as f:
        f.write('\n'.join(hit_ids))
