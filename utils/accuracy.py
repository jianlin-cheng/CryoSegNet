## Code for calculating dice score and jaccard score

def dice_score(target, pred):    
    smooth = 0.0001
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    score = (2 * intersection + smooth) / (union + intersection + smooth)
    return score

def jaccard_score(target, pred):
    smooth = 0.0001
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    score = (intersection + smooth) / (union + smooth)
    return score
