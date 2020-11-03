import numpy as np
from matplotlib import pyplot as plt

interval_num=20
a = list(range(1,interval_num+1))
a = list(map(str,a))
#---------------------------------------------------------------
### Simulation 1
simu108 = [117,67,40,13,8,5,3,3,3,1,0,0,1,0,0,1,0,1,5,32]  # evulate 241  real 241
simu1085 = [123,64,40,9,9,5,3,3,3,1,0,0,1,0,0,1,1,0,5,32]
simu109 = [123,64,40,9,9,5,3,3,3,1,0,0,1,0,0,1,1,0,5,32]

for i in range(18):
	aa = simu108[19-i]
	bb = simu108[19-i-1]
	cc = simu108[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=1 and aa-bb<=0 and bb-cc<=0:
		break
simu108_eva = np.sum(simu108[0:19-i])

for i in range(18):
	aa = simu1085[19-i]
	bb = simu1085[19-i-1]
	cc = simu1085[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=1 and aa-bb<=0 and bb-cc<=0:
		break
simu1085_eva = np.sum(simu1085[0:19-i])

for i in range(18):
	aa = simu109[19-i]
	bb = simu109[19-i-1]
	cc = simu109[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=1 and aa-bb<=0 and bb-cc<=0:
		break
simu109_eva = np.sum(simu109[0:19-i])

print((simu108_eva+simu109_eva+simu1085_eva)/3)


#---------------------------------------------------------------
### Simulation 2
simu208 = [19,9,9,3,20,12,10,7,8,10,16,31,7,12,8,2,4,8,4,1]  # evulate 187  real 200
simu2085 = [20,9,9,6,18,10,11,6,9,11,16,29,7,8,8,5,3,5,8,1]
simu209 = [20,11,7,21,8,14,5,8,6,12,16,20,13,4,9,6,3,3,5,9]

for i in range(18):
	aa = simu208[19-i]
	bb = simu208[19-i-1]
	cc = simu208[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=2 and aa-bb<=0 and bb-cc<=0:
		break
simu208_eva = np.sum(simu208[0:19-i])
print(simu208_eva)
for i in range(18):
	aa = simu2085[19-i]
	bb = simu2085[19-i-1]
	cc = simu2085[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=2 and aa-bb<=0 and bb-cc<=0:
		break
simu2085_eva = np.sum(simu2085[0:19-i])
print(simu2085_eva)
for i in range(18):
	aa = simu209[19-i]
	bb = simu209[19-i-1]
	cc = simu209[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=2 and aa-bb<=0 and bb-cc<=0:
		break
simu209_eva = np.sum(simu209[0:19-i])
print(simu209_eva)
print((simu208_eva+simu209_eva+simu2085_eva)/3)


#---------------------------------------------------------------
### sc-GEM data set
scgem08 = [37,35,23,6,8,1,1,3,2,2,3,1,1,2,5,4,0,3,3,2]
scgem085 = [64,33,11,2,1,2,3,2,2,1,2,1,2,5,3,2,1,0,3,2]
scgem09 = [103,8,1,3,3,2,2,1,1,2,3,2,3,2,0,1,1,3,0,1]

for i in range(18):
	aa = scgem08[19-i]
	bb = scgem08[19-i-1]
	cc = scgem08[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=1 and aa-bb<=0 and bb-cc<=0:
		break
scgem08_eva = np.sum(scgem08[0:20-i])

for i in range(18):
	aa = scgem085[19-i]
	bb = scgem085[19-i-1]
	cc = scgem085[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=1 and aa-bb<=0 and bb-cc<=0:
		break
scgem085_eva = np.sum(scgem085[0:20-i])

for i in range(18):
	aa = scgem09[19-i]
	bb = scgem09[19-i-1]
	cc = scgem09[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=1 and aa-bb<=0 and bb-cc<=0:
		break
scgem09_eva = np.sum(scgem09[0:20-i])

print((scgem08_eva+scgem09_eva+scgem085_eva)/3)


# #---------------------------------------------------------------
### scNMT-seq data set
scnmt08 = [254,57,33,25,14,23,20,45,42,15,9,6,10,9,5,7,5,3,0,2]		# evulate 582, true 584
scnmt085 = [300,48,27,23,20,20,48,36,9,8,12,6,6,6,5,4,3,1,0,2]
scnmt09 = [354,40,23,16,30,44,25,6,9,7,5,6,5,6,4,1,1,0,0,2]

for i in range(18):
	aa = scnmt08[19-i]
	bb = scnmt08[19-i-1]
	cc = scnmt08[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=3 and aa-bb<=0 and bb-cc<=0:
		break
scnmt08_eva = np.sum(scnmt08[0:20-i])

for i in range(18):
	aa = scnmt085[19-i]
	bb = scnmt085[19-i-1]
	cc = scnmt085[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=3 and aa-bb<=0 and bb-cc<=0:
		break
scnmt085_eva = np.sum(scnmt085[0:20-i])

for i in range(18):
	aa = scnmt09[19-i]
	bb = scnmt09[19-i-1]
	cc = scnmt09[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=3 and aa-bb<=0 and bb-cc<=0:
		break
scnmt09_eva = np.sum(scnmt09[0:20-i])

print((scnmt08_eva+scnmt09_eva+scnmt085_eva)/3)


#---------------------------------------------------------------
### PBMC data set
PBMC08 = [1426,29,18,12,12,7,11,7,4,9,4,8,11,7,6,16,8,11,33,280]   # evulate 1649, true 1465
PBMC09 = [1664,18,7,6,8,6,2,4,2,8,4,0,8,6,2,13,10,12,20,119]
PBMC085 = [1547,25,15,5,8,11,2,3,7,4,4,6,3,7,4,14,11,22,25,196]

for i in range(18):
	aa = PBMC08[19-i]
	bb = PBMC08[19-i-1]
	cc = PBMC08[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=10 and aa-bb<=0 and bb-cc<=0:
		break
# print(i)
PBMC08_eva = np.sum(PBMC08[0:20-i])

for i in range(18):
	aa = PBMC09[19-i]
	bb = PBMC09[19-i-1]
	cc = PBMC09[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=10 and aa-bb<=0 and bb-cc<=0:
		break
PBMC09_eva = np.sum(PBMC09[0:20-i])

for i in range(18):
	aa = PBMC085[19-i]
	bb = PBMC085[19-i-1]
	cc = PBMC085[19-i-2]
	if abs(abs(aa-bb)-abs(bb-cc))<=10 and aa-bb<=0 and bb-cc<=0:
		break
PBMC085_eva = np.sum(PBMC085[0:20-i])

print((PBMC08_eva+PBMC09_eva+PBMC085_eva)/3)


# fig = plt.figure(figsize=(10, 6.5))
# plt.plot(a,simu108,'r')
# plt.plot(a,simu1085,'g')
# plt.plot(a,simu109,'b')
# for i in range(20):
# 	plt.plot(a[i], simu108[i], 's', c='r')
# 	plt.plot(a[i], simu1085[i], 's', c='g')
# 	plt.plot(a[i], simu109[i], 's', c='b')

# plt.xticks(fontproperties = 'Arial', size = 18)
# plt.yticks(fontproperties = 'Arial', size = 18)
# plt.xlabel('interval', fontdict={'family' : 'Arial', 'size': 25})
# plt.ylabel('number', fontdict={'family' : 'Arial', 'size': 25})
# plt.title("partial scNMT-seq", font_title)
# fig.savefig('scatter0.eps',dpi=600,format='eps')

fig = plt.figure(figsize=(12, 10))
plt.plot(a,scgem08,'k')
plt.plot(a,scgem085,'k')
plt.plot(a,scgem09,'k')
for i in range(20):
	plt.plot(a[i], scgem08[i], 's', c='r')
	plt.plot(a[i], scgem085[i], 's', c='g')
	plt.plot(a[i], scgem09[i], 's', c='b')

plt.xticks(fontproperties = 'Arial', size = 25)
plt.yticks(fontproperties = 'Arial', size = 25)
plt.xticks(rotation=45)
plt.xlabel('Bin', fontdict={'family' : 'Arial', 'size': 35})
plt.ylabel('Number', fontdict={'family' : 'Arial', 'size': 35})
plt.title('sc-GEM', pad=15, fontdict={'family' : 'Arial', 'size'   : 40})
fig.savefig('scatter1.eps',dpi=600,format='eps')

fig = plt.figure(figsize=(12, 10))
plt.plot(a,scnmt08,'r')
plt.plot(a,scnmt085,'g')
plt.plot(a,scnmt09,'b')
for i in range(20):
	plt.plot(a[i], scnmt08[i], 's', c='r')
	plt.plot(a[i], scnmt085[i], 's', c='g')
	plt.plot(a[i], scnmt09[i], 's', c='b')

plt.xticks(fontproperties = 'Arial', size = 25)
plt.yticks(fontproperties = 'Arial', size = 25)
plt.xticks(rotation=45)
plt.xlabel('Bin', fontdict={'family' : 'Arial', 'size': 35})
plt.ylabel('Number', fontdict={'family' : 'Arial', 'size': 35})
plt.title('scNMT-seq', pad=15, fontdict={'family' : 'Arial', 'size'   : 40})
fig.savefig('scatter2.eps',dpi=600,format='eps')

fig = plt.figure(figsize=(12, 10))
plt.plot(a,PBMC08,'k')
plt.plot(a,PBMC085,'k')
plt.plot(a,PBMC09,'k')
for i in range(20):
	plt.plot(a[i], PBMC08[i], 's', c='r')
	plt.plot(a[i], PBMC085[i], 's', c='g')
	plt.plot(a[i], PBMC09[i], 's', c='b')

plt.xticks(fontproperties = 'Arial', size = 25)
plt.yticks(fontproperties = 'Arial', size = 25)
plt.xticks(rotation=45)
plt.xlabel('Bin', fontdict={'family' : 'Arial', 'size': 35})
plt.ylabel('Number', fontdict={'family' : 'Arial', 'size': 35})
plt.title('PBMC', pad=15, fontdict={'family' : 'Arial', 'size'   : 40})
fig.savefig('scatter3.eps',dpi=600,format='eps')
plt.show()