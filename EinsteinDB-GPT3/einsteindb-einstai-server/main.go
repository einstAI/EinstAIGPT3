package main

import (
	"os"
	"path"

	"git.code.oa.com/gocdb/base/frame"
)

func setValue(p interface{}) {
	switch d := p.(type) {
	case *int64:
		*d = 15
	}

}

func main() {

	configPath := "../etc/" + path.Base(os.Args[0]) + ".conf"
	dapp := NewApp()

	appFrame := frame.NewAppFrame(dapp, configPath)
	if err := appFrame.Init(); err != nil {
		print(err)
		os.Exit(3)
	}
	defer appFrame.Uninit()
	appFrame.Run()

}
