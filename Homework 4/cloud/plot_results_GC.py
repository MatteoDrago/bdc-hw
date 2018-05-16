# Import Packages
import matplotlib.pyplot as plt
import numpy as np

# Load Files
coreset_times = np.load('out/coreset_times.npy')
result_times = np.load('out/result_times.npy')
objs = np.load('out/objs.npy')
k = np.load('out/k.npy')
numBlocks = np.load('out/numBlocks.npy')

########################################
# Plots
########################################

plt.figure(figsize=(17, 3))
interpolation = 'bilinear'

# Objective Function Plot
plt.subplot(1, 3, 1)
plt.title('Objective Function', fontweight='bold')
plt.imshow(objs, interpolation=interpolation, origin='lower')
plt.colorbar()
plt.xticks(range(len(k)), k)
plt.yticks(range(len(numBlocks)), range(np.min(numBlocks), np.max(numBlocks)+1))
plt.xlabel('K', fontweight='bold')
plt.ylabel('Number of Blocks', fontweight='bold')

# Coreset Time Plot
plt.subplot(1, 3, 2)
plt.title('Coreset Time [s]', fontweight='bold')
plt.imshow(coreset_times, interpolation=interpolation, origin='lower')
plt.colorbar()
plt.xticks(range(len(k)), k)
plt.yticks(range(len(numBlocks)), range(np.min(numBlocks), np.max(numBlocks)+1))
plt.xlabel('K', fontweight='bold')
plt.ylabel('Number of Blocks', fontweight='bold')

# Result Time Plot
plt.subplot(1, 3, 3)
plt.title('Result Time [s]', fontweight='bold')
plt.imshow(result_times, interpolation=interpolation, origin='lower')
plt.colorbar()
plt.xticks(range(len(k)), k)
plt.yticks(range(len(numBlocks)), range(np.min(numBlocks), np.max(numBlocks)+1))
plt.xlabel('K', fontweight='bold')
plt.ylabel('Number of Blocks', fontweight='bold')

plt.show()

numBlocks_list = [i for i in range(np.min(numBlocks), np.max(numBlocks)+1)]
k_list = [i for i in k]

plt.figure(figsize=(17,3))

# Objective Function Plot
plt.subplot(1, 3, 1)
for i in range(len(numBlocks)):
    plt.plot(k_list, objs[i,:], label='numBlock='+str(numBlocks_list[i]))
plt.legend(loc=1, prop={'size':8})
plt.grid(True)
plt.xlabel('K', fontweight='bold')
plt.ylabel('Objective Functions', fontweight='bold')

# Coreset Time Plot
plt.subplot(1, 3, 2)
for i in range(len(numBlocks)):
    plt.plot(k_list, coreset_times[i,:], label='numBlock='+str(numBlocks_list[i]))
plt.legend(loc=1, prop={'size':8})
plt.grid(True)
plt.xlabel('K', fontweight='bold')
plt.ylabel('Coreset Time [s]', fontweight='bold')

# Result Time Plot
plt.subplot(1, 3, 3)
for i in range(len(numBlocks)):
    plt.plot(k_list, result_times[i,:], label='numBlock='+str(numBlocks_list[i]))
plt.legend(loc=2, prop={'size':8})
plt.grid(True)
plt.xlabel('K', fontweight='bold')
plt.ylabel('Result Time [s]', fontweight='bold')

plt.show()