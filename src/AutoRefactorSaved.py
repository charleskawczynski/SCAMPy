# This file was saved immediately after work, Monday 7/15/2019
import fileinput
import os

ext = ['.py']
D_old = {}
D_new = {}
D_old['q'] = ('W', 'Area', 'q_tot', 'θ_liq', 'q_rai')
D_new['q'] = ('w', 'a'   , 'q_tot', 'θ_liq', 'q_rai')

D_old['tmp'] = ('q_liq', 'T', 'B')
D_new['tmp'] = ('q_liq', 'T', 'B')
D_single_sd = {}
D_single_sd['q_liq']                  = False
D_single_sd['T']                      = False
D_single_sd['B']                      = False

# Prognostic
structs = ('UpdVar',)
indexes = ('i',)

root_folder = os.path.dirname(__file__)
all_files = [root+os.sep+f for root, dirs, files in os.walk(root_folder) for f in files if any([f.endswith(x) for x in ext])]
all_files = [x for x in all_files if not x.endswith(__file__)]

# print('\n'.join(all_files))
# modify_code = True
modify_code = True

L_old = []
L_new = []

for k_old, k_new in zip(D_old, D_new):
  k = k_new
  for v_old, v_new in zip(D_old[k], D_new[k]):
    v = v_old
    if k=='tmp':
      name_index_only = D_single_sd[v]
    else:
      name_index_only = False
    for s, i in zip(structs, indexes):

      close_index = "']" if name_index_only else "', "+i+"]"
      for fun_call in ("Dual", "Cut", "DualCut", "Mid"):
        L_old += [s+"."+v_old+".values[i]."+fun_call+"("]
        L_new += [k+"['"+v_new+close_index+"."+fun_call+"("]

      L_old += [s+"."+v_old+".set_bcs(grid)"]
      L_new += ["for i in i_uds: "+k+"['"+v_new+"', "+i+"].apply_bc(grid, 0.0)"]

      L_old += [s+"."+v_old+".values[i]["]
      L_new += [k+"['"+v_new+close_index+"["]

      L_old += [s+"."+v_old+"[i]["]
      L_new += [k+"['"+v_new+close_index+"["]

      # Args
      L_old += [s+"."+v_old+","]
      L_new += [k+"['"+v_new+close_index+","]
      L_old += [s+"."+v_old+" "]
      L_new += [k+"['"+v_new+close_index+" "]
      L_old += [s+"."+v_old+".values"]
      L_new += [k+"['"+v_new+close_index]


if modify_code:
  for f in all_files:
    with open(f, 'r', encoding="utf-8") as file:
      filedata = file.read()
    for s_old, s_new in zip(L_old, L_new):
      filedata = filedata.replace(s_old, s_new)
    with open(f, 'w', encoding="utf-8") as file:
      file.write(filedata)
else:
  for s_old, s_new in zip(L_old, L_new):
    print('s_old, s_new = ', s_old, '\t\t\t', s_new)

