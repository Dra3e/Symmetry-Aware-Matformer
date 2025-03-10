def LossProperty(actual_Y, predicted_Y):
    if len(actual_Y) == len(predicted_Y) and actual_Y.shape == predicted_Y.shape:
        loss = 0
        for i in actual_Y:
            loss += (actual_Y[i] - predicted_Y[i]) ** 2
        loss = loss / len(actual_Y)
        return loss
    else:
        raise Exception("Inputs are not the same size!")

def LossContrastive(actual_F, predicted_F, margin, distance_function):
    raise Exception("Not Implemented")

def LossFactor(loss, factor):
    raise Exception("Not Implemented")

def LossTotal(loss, factor):
    raise Exception("Not Implemented")

def EuclidianDistance(point_A, point_B):
    raise Exception("Not Implemented")

def CosineSimilarity(point_A, point_B):
    raise Exception("Not Implemented")