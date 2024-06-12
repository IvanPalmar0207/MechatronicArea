//If you want to see the classes youmodel has you gotta do the next:

print(model.names)

If you need to add only a few classes to your report then you gotta do the next

    results = model.track(frame,persist = True, classes = (0,1,2,3,4,5,6,7,8))
