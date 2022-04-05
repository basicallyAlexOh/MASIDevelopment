"""Scripts for QA pipeline"""
import os
import sys
import shutil

def get_passed_qa(clip_passed_qa_dir, src_dir, dst_dir, passed_qa_list):
    '''writes list of scan IDs that passed QA and copy corresponding raw images to raw_dir'''
    passed = []
    with open(passed_qa_list, 'w+') as f:
        for clip_name in os.listdir(clip_passed_qa_dir):
            scanid = clip_name.split("_")[0]
            if scanid not in passed:
                # record scanid that passed qa
                f.write(scanid + '\n')
                # copy raw CT to raw_dir
                src = os.path.join(src_dir, f"{scanid}.nii.gz")
                dst = os.path.join(dst_dir, f"{scanid}.nii.gz")
                shutil.copyfile(src, dst)
                passed.append(scanid)

if __name__=="__main__":
    print(sys.argv)
    get_passed_qa(*sys.argv[1:])