#!/usr/bin/env python

import sys, os
from glob import glob
import time
import subprocess
import argparse

DO_PANDOC = False
DO_PANDOC_TOP = True

SELF_CONTAINED = True

ORDER_NEWEST = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The path to generate report for")
    args = parser.parse_args()

    path = args.path

    # kill trailing slash
    if not os.path.split(path)[1]:
        path = os.path.split(path)[0]

    tstart = time.time()
    topfile_name = os.path.join(path, "summary.md")
    topfile = open(topfile_name, 'w')
    print("Create file %s"%topfile_name)
    topfile.write("%%Summary for %s\n\n" % os.path.basename(path))
    topfile.write("## Summary: %s\n\n"%os.path.basename(path))
    topfile.write("Report time: %d\n\n"%tstart)

    if ORDER_NEWEST:
        topfile.write("Ordering by newest\n\n")
        paths_and_times = []
        for f in glob(os.path.join(path,"*","losses.png")):
            paths_and_times.append((os.stat(f).st_mtime, os.path.dirname(f)))
        paths_and_times.sort(reverse=True)
        paths = [p for _,p in paths_and_times]
    else:
        paths = list(glob(os.path.join(path,"*")))
    #for d in sorted(glob(os.path.join(path,"*"))):

    options = set()
    for p in paths:
        for i in os.path.basename(p).split("_")[1:]:
            options.add(i)
    options = sorted(options)
    for opt in options:
        topfile.write('<input type="checkbox" class="hp-filter" value="%s" checked>%s</input>'%(opt.replace(".","_"),opt))
    topfile.write("\n\n")
    for sopt in ('training', 'abort', 'complete', 'other'):
        topfile.write('<input type="checkbox" class="hp-filter" value="%s" checked>%s</input>'%(sopt,sopt))
    topfile.write("\n\n")
    for vopt in ('novideo', 'video'):
        topfile.write('<input type="checkbox" class="hp-filter" value="%s" checked>%s</input>'%(vopt,vopt))
    topfile.write("\n\n")

    for d in paths:
        d0 = os.path.basename(d)
        if not os.path.isdir(d): continue
        try:
            slurm_id = open(os.path.join(d, "slurm_job")).read().strip()
            status = [l.strip() for l in open(os.path.join(d, "status")).readlines()]
        #except FileNotFoundError:
        #    continue
        except IOError:
            continue

        current_status = 'other'
        status_str = ''.join(status)
        if 'complete' in status_str:
            current_status = 'complete'
        elif 'abort' in status_str:
            current_status = 'abort'
        elif status[-1] == 'training':
            current_status = 'training'
        
        last_update = 0
        for imgfile in glob(os.path.join(d, "*.png")):
            last_update = max(last_update, os.stat(imgfile).st_mtime)

        dirfile_name = os.path.join(d, 'summary.md')
        dirfile = open(dirfile_name, 'w')
        print("Create file %s"%dirfile_name)

        dirfile.write("%%Summary for %s\n\n" % d0)

        video_links = []
        vclass = 'novideo'
        for i,vf in enumerate(glob(os.path.join(path,"video_*_%s.json"%d0))):
            video_links.append('<a href="render.html?seq=%s">video %d</a>  '%(os.path.basename(vf), i))
            vclass = 'video'

        classes_str = ' '.join([c.replace(".","_") for c in d0.split("_")])
        topfile.write('<div id="%s" class="trial %s %s %s">'%(d0, current_status, classes_str, vclass))
        topfile.write("### %s\n\n" % d0)
        dirfile.write("## %s\n\n" % d0)

        #topfile.write("[Details](%s)\n\n"%os.path.join(d0, 'summary.html'))
        for vl in video_links:
            topfile.write(vl)
        topfile.write("\n\n")

        def fmt_status(status):
            for i in range(len(status)):
                if 'abort' in status[i]:
                    status[i] = "**<font color='#D817FF'>%s</font>**"%status[i]
                if 'exception' in status[i]:
                    status[i] = "<font color='red'>%s</font>"%status[i]
            if status[-1] == 'training':
                status[-1] = "<font color='green'>**TRAINING**</font>"
            return " → ".join(status)
        topfile.write("Status: %s\n\n" % fmt_status(status))
        dirfile.write("Status: %s\n\n" % status)

        topfile.write("Slurm ID: %s\n\n" % slurm_id)
        dirfile.write("Slurm ID: %s\n\n" % slurm_id)
        
        topfile.write("Last update: %s\n\n" % last_update)
        dirfile.write("Last update: %s\n\n" % last_update)

        dirfile.write("### Losses\n\n")
        loss_file = os.path.join(d, "losses.png")
        if os.path.exists(loss_file):
            topfile.write("![](%s)\n"%loss_file)
            dirfile.write("![](%s)\n"%loss_file)
        dirfile.write("### Rigidity")
        for i,rfile in enumerate(sorted(glob(os.path.join(d, 'rigidity_e*.png')), reverse=True)[::2]):
            #fpath, fbase = os.path.split(rfile)
            #rfile = os.path.join(os.path.basename(fpath), fbase)
            if i == 0:
                topfile.write("![](%s)\n"%rfile)
            dirfile.write("![](%s)\n"%rfile)
        dirfile.write("### Poses\n\n")
        for i,pfile in enumerate(sorted(glob(os.path.join(d, 'poses_e*.png')), reverse=True)[::2]):
            #fpath, fbase = os.path.split(pfile)
            #rfile = os.path.join(os.path.basename(fpath), fbase)
            if i == 0:
                topfile.write("![](%s)\n"%pfile)
            dirfile.write("![](%s)\n"%pfile)
        topfile.write("</div>\n")
        topfile.write("\n\n")
        dirfile.write("\n\n")

        dirfile.close()

        if DO_PANDOC:
            print("Generating html file...")
            sys.stdout.flush()
            html_name = os.path.join(d, "summary.html")
            subprocess.run(["pandoc", "--self-contained", "-o", html_name, dirfile_name])

    topfile.write('<script src="https://code.jquery.com/jquery-3.3.1.min.js" crossorigin="anonymous"></script>')
    topfile.write("""<script>
    hidden = new Set();
$(".hp-filter").click(function () {
    if (hidden.has(this.value)) {
        hidden.delete(this.value);
    } else {
        hidden.add(this.value);
    }
    //$("."+this.value).toggle();
    $(".trial").show();
    for (let i of hidden) {
        $("."+i).hide();
    }
    });
</script>
""")

    topfile.close()
    if DO_PANDOC or DO_PANDOC_TOP:
        print("Generating top-level html file...")
        sys.stdout.flush()
        html_name = os.path.join(path, "summary.html")
        cmd = ['pandoc']
        if SELF_CONTAINED:
            cmd.append("--self-contained")
        cmd += ['-o', html_name, topfile_name]
        subprocess.run(cmd)
    print("Done.")
