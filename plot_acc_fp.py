import matplotlib.pyplot as plt

fprate_2ch = {2: 13.796833558128379, 3: 7.939187660109893, 4: 5.853572430773688, 5: 4.8392790868191655} 
acc_2ch = {2: 0.13043478260869565, 3: 0.13043478260869565, 4: 0.13043478260869565, 5: 0.13043478260869565}

fprate_6ch = {2: 0.5398546545160918, 3: 0.24916368669973468, 4: 0.0830545622332449, 5: 0.0} 
acc_6ch = {2: 0.14285714285714285, 3: 0.14285714285714285, 4: 0.14285714285714285, 5: 0.14285714285714285}

fprate_8ch = {2: 91.52612758103588, 3: 57.39070250317222, 4: 37.789825816126424, 5: 27.11731456915446} 
acc_8ch = {2: 0.9047619047619048, 3: 0.7142857142857143, 4: 0.5714285714285714, 5: 0.47619047619047616}

fprate_9ch = {2: 20.223785903795132, 3: 7.059637789825816, 4: 2.5331641481139693, 5: 1.0797093090321837} 
acc_9ch = {2: 0.47619047619047616, 3: 0.42857142857142855, 4: 0.3333333333333333, 5: 0.3333333333333333}

k_range = range(2,6)
fig, ax = plt.subplots(1, 2)
ax[0].plot(k_range, list(fprate_2ch.values()), label='ch2') 
ax[0].plot(k_range, list(fprate_6ch.values()), label='ch6') 
ax[0].plot(k_range, list(fprate_8ch.values()), label='ch8') 
ax[0].plot(k_range, list(fprate_9ch.values()), label='ch9') 
ax[0].set_ylabel('False Positive Rate (FP/hr)')
ax[0].set_xlabel('k')
plt.xticks(k_range)
plt.xlabel('k')

ax[1].plot(k_range, list(acc_2ch.values()), label='ch2')
ax[1].plot(k_range, list(acc_6ch.values()), label='ch6')
ax[1].plot(k_range, list(acc_8ch.values()), label='ch8')
ax[1].plot(k_range, list(acc_9ch.values()), label='ch9')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('k')
plt.xticks(k_range)
plt.xlabel('k')

plt.setp(ax, xticks=k_range)
plt.suptitle('Model performance')

ch2_k = [0.6*acc_2ch[k] - 0.4*fprate_2ch[k] for i, k in enumerate(fprate_2ch)]
ch6_k = [0.6*acc_6ch[k] - 0.4*fprate_6ch[k] for i, k in enumerate(fprate_6ch)]
ch8_k = [0.6*acc_8ch[k] - 0.4*fprate_8ch[k] for i, k in enumerate(fprate_8ch)]
ch9_k = [0.6*acc_9ch[k] - 0.4*fprate_9ch[k] for i, k in enumerate(fprate_9ch)]

print(ch2_k, ch6_k, ch8_k, ch9_k)


plt.legend()
plt.show()
