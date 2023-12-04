from trackers.deep_sort.utils.parser import get_config
from trackers.deep_sort.deep_sort import DeepSort

def create_tracker(tracker_type, model_type=None):
    if tracker_type == 'deepsort':
        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file("trackers/deep_sort/configs/deep_sort.yaml")
        model_type = model_type if model_type is not None else cfg.DEEPSORT.MODEL_TYPE
        deepsort = DeepSort(model_type,
                            max_dist=cfg.DEEPSORT.MAX_DIST,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        
        return deepsort

    else:
        print('No such tracker')
        exit()