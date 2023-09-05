import cv2 
import glob

files = sorted(glob.glob('0F598DC2-29E5-41F7-B374-4A3F5E27216D20230724/'+'*.JPG'))
i=0
for img_fp in files:
    print("Process the image: ", img_fp)
    img_ori = cv2.imread(img_fp)
    img_ori = cv2.flip(cv2.transpose(img_ori), 1)
    name = img_fp.rsplit('/',1)[-1][:-4]
    cv2.imwrite('img_100/{}.JPG'.format(name), img_ori)
    i+=1
    if i >100: break

