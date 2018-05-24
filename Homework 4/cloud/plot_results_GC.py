# Import Packages
import matplotlib.pyplot as plt
import numpy as np

# Load Files
coreset_times = np.load('out/GC/coreset_times.npy')
result_times = np.load('out/GC/result_times.npy')
objs = np.load('out/GC/objs.npy')
k = np.load('out/GC/k.npy')
numBlocks = np.load('out/GC/numBlocks.npy')

########################################
# Plots
########################################

plt.figure(figsize=(10, 10))
interpolation = None

# Objective Function Plot
plt.subplot(3, 1, 1)
plt.title('Objective Function', fontweight='bold')
plt.imshow(objs, interpolation=interpolation, origin='lower', aspect='auto')
plt.colorbar()
plt.xticks(range(len(k))[0::5], k[0::5])
plt.yticks(range(len(numBlocks)), range(np.min(numBlocks), np.max(numBlocks)+1))
plt.xlabel('K', fontweight='bold')
plt.ylabel('Number of Blocks', fontweight='bold')
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                wspace=0.2, hspace=0.4)

# Coreset Time Plot
plt.subplot(3, 1, 2)
plt.title('Coreset Time [s]', fontweight='bold')
plt.imshow(coreset_times, interpolation=interpolation, origin='lower', aspect='auto')
plt.colorbar()
plt.xticks(range(len(k))[0::5], k[0::5])
plt.yticks(range(len(numBlocks)), range(np.min(numBlocks), np.max(numBlocks)+1))
plt.xlabel('K', fontweight='bold')
plt.ylabel('Number of Blocks', fontweight='bold')
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                wspace=0.2, hspace=0.4)

# Result Time Plot
plt.subplot(3, 1, 3)
plt.title('Result Time [s]', fontweight='bold')
plt.imshow(result_times, interpolation=interpolation, origin='lower', aspect='auto')
plt.colorbar()
plt.xticks(range(len(k))[0::5], k[0::5])
plt.yticks(range(len(numBlocks)), range(np.min(numBlocks), np.max(numBlocks)+1))
plt.xlabel('K', fontweight='bold')
plt.ylabel('Number of Blocks', fontweight='bold')
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                wspace=0.2, hspace=0.4)
plt.savefig('plots/GC/1.eps')
plt.show()

numBlocks_list = [i for i in range(np.min(numBlocks), np.max(numBlocks)+1)]
k_list = [i for i in k]

plt.figure(figsize=(10, 10))

# Objective Function Plot
plt.subplot(3, 1, 1)
for i in range(len(numBlocks)):
    plt.plot(k_list, objs[i,:], label='numBlock='+str(numBlocks_list[i]))
plt.legend(loc=1, prop={'size':8})
plt.grid(True)
plt.xticks(range(len(k))[0::5], k[0::5])
plt.xlim(np.min(k), np.max(k))
plt.xlabel('K', fontweight='bold')
plt.ylabel('Objective Functions', fontweight='bold')
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                wspace=0.2, hspace=0.4)

# Coreset Time Plot
plt.subplot(3, 1, 2)
for i in range(len(numBlocks)):
    plt.plot(k_list, coreset_times[i,:], label='numBlock='+str(numBlocks_list[i]))
plt.legend(loc=2, prop={'size':8})
plt.grid(True)
plt.xticks(range(len(k))[0::5], k[0::5])
plt.xlim(np.min(k), np.max(k))
plt.xlabel('K', fontweight='bold')
plt.ylabel('Coreset Time [s]', fontweight='bold')
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                wspace=0.2, hspace=0.4)

# Result Time Plot
plt.subplot(3, 1, 3)
for i in range(len(numBlocks)):
    plt.plot(k_list, result_times[i,:], label='numBlock='+str(numBlocks_list[i]))
plt.legend(loc=2, prop={'size':8})
plt.grid(True)
plt.xticks(range(len(k))[0::5], k[0::5])
plt.xlim(np.min(k), np.max(k))
plt.xlabel('K', fontweight='bold')
plt.ylabel('Result Time [s]', fontweight='bold')
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                wspace=0.2, hspace=0.4)
plt.savefig('plots/GC/2.eps')
plt.show()