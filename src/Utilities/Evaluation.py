

from Data import Data


class Evaluator:

    def __init__(self):
        self.HR = list()
        self.NDCG = list()

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0

    def evaluate(self, model, K):

        data = Data()
        train, test = data.Get_data()

        for i in range(len(test)):

            rating = test[i]
            u = rating[0]

            # taking 99 randome untested conditions by that drug
            count = 0
            drugs = []
            while(count != 99):
                j = random.randint(0, 3435)
                if (u, j) in train.keys():
                    continue
                drugs.append(j)
                count += 1

            gtdrug = rating[1]
            drugs.append(gtdrug)

            # Get prediction scores
            map_drug_score = {}
            medical_conditions = np.full(len(drugs), u, dtype='int32')
            predictions = model.predict([medical_conditions, np.array(drugs)],
                                        batch_size=64, verbose=0)

            for i in range(len(drugs)):
                drug = drugs[i]
                map_drug_score[drug] = predictions[i]

            drugs.pop()

            ranklist = heapq.nlargest(
                K, map_drug_score, key=map_drug_score.get)
            hr = self.getHitRatio(ranklist, gtdrug)
            ndcg = self.getNDCG(ranklist, gtdrug)

            self.HR.append(hr)
            self.NDCG.append(ndcg)

        return (self.HR, self.NDCG)
