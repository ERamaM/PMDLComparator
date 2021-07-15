import os
import sys
import shutil


myPath = os.path.dirname(getBundlePath())
if not myPath in sys.path: sys.path.append(myPath)
popup(myPath)
for file in os.listdir(os.path.join(myPath, "declare_miner_files")):
    if "train_val" in file:

        wait("1626165107316.png", 5)

        click("1626165165354.png")

        wait("1626165206839.png", 60)

        click("1626165223245.png")

        wait("1626165301029.png")

        click("1626165315869.png")

        wait("1626166422909.png", 2)

        paste(os.path.join(myPath, "declare_miner_files", file))

        type(Key.ENTER)

        click("1626166517594.png")

        while exists("1626170767267.png"):
            sleep(1)

        click("1626166592033.png")

        click("1626166613276.png")


        find("1626166638426.png")

        click(Region(572,346,21,21))

        find("1626169671140.png")

        click(Region(571,647,24,19))

        find("1626167119226.png")

        doubleClick(Region(398,328,73,19))

        type("10")

        click("1626170342569.png")

        while exists("1626170388907.png"):
            sleep(1)

        click("1626167295392.png")

        wait("1626188200917.png", 5)

        paste(os.path.join(myPath, "declare_models", file.replace(".xes.gz", ".decl")))

        click("1626167481295.png")
  

        wait("1626167674717.png", 5)

        click("1626167694553.png")

        click("1626167520853.png")

        wait("1626188246205.png", 60)

        click("1626167547549.png")

        click("1626167566956.png")

        wait("1626188307438.png", 5)

        paste(os.path.join(myPath, "declare_models", file.replace(".xes.gz", ".decl")))

        type(Key.ENTER)

        click("1626167751784.png")
    
        click("1626167783067.png")

        click("1626167926537.png")

        paste(os.path.join(myPath, "declare_miner_files", file.replace("train_", "")))

        type(Key.ENTER)

        click("1626168785721.png")

        click("1626167949616.png")

        wait("1626167973383.png", 10)

        click("1626168344639.png")

        click("1626168353825.png")

        wait(0.5)

        click("1626168384438.png")

        wait(1.5)

        focusWindow = App.focusedWindow()
        regionImage = capture(focusWindow)
        shutil.move(regionImage, os.path.join(myPath, "declare_rules", file.replace("train_val_", "").replace(".xes.gz", ".png")))
