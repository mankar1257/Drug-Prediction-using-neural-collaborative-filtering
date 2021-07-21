

def Get_model(num_medical_conditions, num_drugs, layers=[16, 8], reg_layers=[0, 0]):

    assert len(layers) == len(reg_layers)

    num_layer = len(layers)  # Number of layers in the MLP

    medical_condition_input = Input(
        shape=(1,), dtype='int32', name='user_input')
    drug_input = Input(shape=(1,), dtype='int32', name='item_input')

    MLP_Embedding_Medical_Conditions = Embedding(input_dim=num_medical_conditions, output_dim=int(layers[0]/2),
                                                 name='medical_condition_embedding',
                                                 W_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Drugs = Embedding(input_dim=num_drugs, output_dim=int(layers[0]/2),
                                    name='drug_embedding',
                                    W_regularizer=l2(reg_layers[0]), input_length=1)

    medical_condition_latent = Flatten()(MLP_Embedding_Medical_Conditions(medical_condition_input)
                                         )                                      # flattening embedding for user
    # flattening embedding for items
    drug_latent = Flatten()(MLP_Embedding_Drugs(drug_input))

    # forming the 0th layer of NN by concatinating the user and items flatten layer
    vector = keras.layers.concatenate([medical_condition_latent, drug_latent])

    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], W_regularizer=l2(
            reg_layers[idx]), activation='relu', name='layer%d' % idx)
        vector = layer(vector)
        #layer1 = Dropout(0.25)
        #vector = layer1(vector)

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid',
                       init='lecun_uniform', name='prediction')(vector)

    model = Model(input=[medical_condition_input, drug_input],
                  output=prediction)

    return model


def Get_train_instances(train, num_negatives):

    medical_condition_input, drug_input, labels = [], [], []
    num_medical_conditions = train.shape[0]

    for (u, i) in train.keys():

        # positive instance
        medical_condition_input.append(u)
        drug_input.append(i)
        labels.append(1)

        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_drugs)

            while ((u, j) in train.keys()):
                j = np.random.randint(num_drugs)
            medical_condition_input.append(u)
            drug_input.append(j)
            labels.append(0)

    return medical_condition_input, drug_input, labels
