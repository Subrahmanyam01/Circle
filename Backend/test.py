import cv2
from ts import YoloEnex
if True:
            vs = cv2.VideoCapture('video.mp4')
            er = YoloEnex(vertical=False, line_position=600, skip_rate=5, show_line=True, show_bbox=True)
            while True:
                    (grabbed, frame) = vs.read()
                    c=0
                    if grabbed:
                            #c+=1
                            #if c%5==0:
                                frame,totalUpLeft,totalDownRight,totalInBetween,total_entry_count,classes,lb,rb,left, right, = er.inference(frame)
                                """"
                                for i in range(len(rb)):
                                    bbox=rb[i]
                                    cv2.rectangle(
                                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
                                    )
                                """
                                print(totalUpLeft,totalDownRight)
                                cv2.imshow('test',cv2.resize(frame,(720,480)))
                           # else:
                           #     cv2.imshow('test',cv2.resize(frame,(720,480)))
                                if cv2.waitKey(25) == ord("q"):
                                        break
            cv2.destroyAllWindows()
            vs.release()
