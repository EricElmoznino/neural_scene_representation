import torch


def rep_using_rooms(scene_data, model, vae=False, average_over_room=True):
    prev_room = None
    representations = []
    for room, image, viewpoint in scene_data:
        if room != prev_room:
            n_frames = 1
        with torch.no_grad():
            if not vae:
                phi = model(image.unsqueeze(0)).squeeze(0)
            else:
                phi = model(image.unsqueeze(0), viewpoint.unsqueeze(0)).squeeze(0)
        phi = phi.view(-1)
        if average_over_room and room == prev_room:
            r += phi
            representations.append(r / n_frames)
        else:
            r = phi
            representations.append(r)
        prev_room = room
        n_frames += 1
    return representations



def rep_using_bayesian_surprise(scene_data, model):
    pass