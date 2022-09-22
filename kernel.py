"""
Find NLST scans with soft kernel and copy over
"""
import os
import sys
import pandas as pd
import paramiko
import argparse
from tqdm import tqdm

def main(src_dir, dst_dir, sample, auth, log):
    # remote copy via ssh

    ssh = paramiko.SSHClient()
    ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    ssh.connect(auth['server'], username=auth['username'], password=auth['password'])
    sftp = ssh.open_sftp()
    print(ssh.get_transport().getpeername())
    # sftp.put(localpath, remotepath)

    failed = []
    scanids = pd.read_csv(sample)['series_uid'].tolist()
    for scanid in tqdm(scanids):
        src = os.path.join(src_dir, f"{scanid}.nii.gz")
        dst = os.path.join(dst_dir, f"{scanid}.nii.gz")
        # print(src)
        try:
            sftp.get(src, dst)
        except Exception as e:
            failed.append(scanid)
            print(src)
            print(e)

    with open(log, 'w') as f:
        f.write(",".join(failed))

    sftp.close()
    ssh.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=str, default='masi-41.vuds.vanderbilt.edu')
    parser.add_argument('--username', type=str, default='litz')
    parser.add_argument('--password', type=str)

    args = parser.parse_args()
    src_dir = "/local_storage/Data/NLST/NIfTI/T0_all"
    dst_dir= "/home/litz/data/NLST/T0_softkernel"
    sample= "/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/nlst_sample.csv"
    log = "/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/tmp/kernel_log.csv"
    auth={"server": args.server, "username": args.username, "password": args.password}

    main(src_dir, dst_dir, sample, auth, log)