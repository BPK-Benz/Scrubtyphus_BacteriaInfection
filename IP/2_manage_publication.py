import os
import json
import utils.make_coco as make_coco


def load_coco(coco_path):
    with open(coco_path) as file:
        coco = json.load(file)
    return coco

def save_coco(coco_path, coco):
    with open(coco_path, 'w') as outfile:
        json.dump(coco, outfile)

def create_coco():
    return {
        "info": make_coco.add_info(),
        "licenses": make_coco.add_licenses(),
        "categories": [],
        "images": [],
        "annotations": [],
    }


def condition(annotation=None):
    if not annotation:
        return [
            {
                "supercategory": 'Cell', # cell, nucleus, properties of objects
                "id": 1, # order of class
                "name": 'Infected_cells', # class name
            },
            {
                "supercategory": 'Cell',
                "id": 2,
                "name": 'Uninfected_cells'
            },
            {
                "supercategory": 'Cell',
                "id": 3,
                "name": 'Irrelevant_cells',
            },
        ]
    else:

        channel = annotation['channel']
        divide = annotation['divide']
        border = annotation['border']
        infect = annotation['infect']

       
        if channel == 'nucleus':
            if divide: return 3
            elif border: return 3
            elif infect == "non-infected": return 2
            else: return 1


if __name__ == "__main__":


    data = {
        'train': {
            'src':[
                'data_old/output_coco/S1/Plate_03/Split_scramble',
                'data_old/output_coco/S1/Plate_04/Split_scramble',
                'data_old/output_coco/S1/Plate_07/Split_scramble',
                'data_old/output_coco/S1/Plate_03/Split_testing',
                'data_old/output_coco/S1/Plate_04/Split_testing',
                'data_old/output_coco/S1/Plate_07/Split_testing',
            ],
            'dst':'InfectTotal_TrainNuc_3class.json',
        },
        'test': {
            'src':[
                'data_old/output_coco/S1/Plate_03/Scramble',
                'data_old/output_coco/S1/Plate_04/Scramble',
                'data_old/output_coco/S1/Plate_07/Scramble',
                'data_old/output_coco/S1/Plate_03/Testing',
                'data_old/output_coco/S1/Plate_04/Testing',
                'data_old/output_coco/S1/Plate_07/Testing',
            ],
            'dst':'InfectTotal_TestNuc_3class.json',
        },
    }

    for key in data:

        all_coco = create_coco()
        all_coco['categories'] = condition()
        print(all_coco['categories'])

        count_folder = 1
        count_image = 1
        count_annotation = 1

        folders = data[key]['src']
        for folder in folders:

            print('[ {} : {} of {} | {} ]'.format(key, count_folder, len(folders), folder))
            count_folder += 1

            files = sorted(os.listdir(folder))
            for file in files:

                coco_path = os.path.join(folder, file)
                coco = load_coco(coco_path)

                images = coco['images']
                annotations = coco['annotations']
                new_image_ids = {}

                for i in range(len(images)):
                    new_image_ids[images[i]['id']] = count_image                    
                    images[i]['id'] = count_image
                    images[i]['file_name'] = images[i]['file_name'].replace('F:\\cellLabel-main\\', '').replace('\\','/')
                    images[i]['file_name'] = images[i]['file_name'].replace('/share/NAS/Benz_Cell/cellLabel-main/', '')
                    count_image += 1

                new_annotations = []
                for a in range(len(annotations)):
                    label = condition(annotations[a])
                    if not label: continue
                    coco_object = {
                        'category_id': label,
                        'image_id': new_image_ids[annotations[a]['image_id']],
                        'id': count_annotation,
                        'iscrowd': 0,
                        'channel': annotations[a]['channel'],
                        'border': annotations[a]['border'],
                        'divide': annotations[a]['divide'],
                        'infect': annotations[a]['infect'],
                        'area': annotations[a]['area'],
                        'bbox': annotations[a]['bbox'],
                        'segmentation': annotations[a]['segmentation'],
                    }
                    new_annotations.append(coco_object)
                    count_annotation += 1

                all_coco['images'] += images
                all_coco['annotations'] += new_annotations

        all_coco_path = data[key]['dst']
        all_coco_path = 'Coco_File/'+all_coco_path
        save_coco(all_coco_path, all_coco)

