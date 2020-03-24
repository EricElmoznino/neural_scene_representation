from argparse import ArgumentParser
import json
import pandas as pd
import xmltodict
import boto3


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
                         aws_access_key_id=credentials['aws_access_key_id'],
                         aws_secret_access_key=credentials['aws_secret_access_key'],
                         region_name='us-east-1',
                         endpoint_url=mturk_environment['endpoint'])

    return mturk


def read_assignment(assignment):
    xml_responses = xmltodict.parse(assignment['Answer'])
    response = {}
    if type(xml_responses['QuestionFormAnswers']['Answer']) is not list:
        xml_responses['QuestionFormAnswers']['Answer'] = [xml_responses['QuestionFormAnswers']['Answer']]
    for answer_field in xml_responses['QuestionFormAnswers']['Answer']:
        field = str(answer_field['QuestionIdentifier'])
        answer = str(answer_field['FreeText'])
        response[field] = answer
    return response


if __name__ == '__main__':
    parser = ArgumentParser(description='Automatically retrieve hit results')
    parser.add_argument('--hit_ids', required=True, type=str, help='filename of hit ids to retrieve')
    parser.add_argument('--production', action='store_true', help='retrieve hits from production environment')
    args = parser.parse_args()

    with open('credentials.json', 'r') as f:
        credentials = json.loads(f.read())
    mturk = setup_mturk_connection(credentials, production=args.production)

    with open('created_hits/' + args.hit_ids, 'r') as f:
        hit_ids = f.read().split('\n')

    responses = []
    num_hits_completed = 0
    for hit_id in hit_ids:
        worker_results = mturk.list_assignments_for_hit(HITId=hit_id)
        if worker_results['NumResults'] < 0:
            continue

        num_hits_completed += 1
        for assignment in worker_results['Assignments']:
            responses.append(read_assignment(assignment))

    responses = pd.DataFrame(responses)
    responses.to_csv('hit_results/' + args.hit_ids.replace('.txt', '.csv'), index=False)
    print('Obtained results for {} completed hits'.format(num_hits_completed))
