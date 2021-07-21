''' functionalities for aws s3 interaction '''


def Upload_to_S3(s3_prefix):

    traindata_s3_prefix = '{}/data/train'.format(s3_prefix)
    testdata_s3_prefix = '{}/data/test'.format(s3_prefix)

    train_s3 = sagemaker.Session().upload_data(
        path='./data/train/', key_prefix=traindata_s3_prefix)
    test_s3 = sagemaker.Session().upload_data(
        path='./data/test/', key_prefix=testdata_s3_prefix)

    inputs = {'train': train_s3, 'test': test_s3}

    return inputs


def Download():
    # if needed
    return 0
