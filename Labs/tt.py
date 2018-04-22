import cv2

query = cv2.imread("7.jpg")
a_r = 200.0/query.shape[0] # scale between old and new width
out  = cv2.resize(query, (0, 0), fx=a_r, fy=a_r)
cv2.imwrite("query.jpg", out)

