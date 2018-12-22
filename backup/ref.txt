package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"
)

// example1 := "Doug saw Jane."
// example2 := "Jane saw Spot."
// example3 := "Spot saw Doug."

// there should be 5 vectors here "Jane0 saw1 Doug2 Spot3 .4"

// X is inputs
var X = [][]float64{
	{0, 0, 1, 0, 0},
	{0, 1, 0, 0, 0},
	{0, 0, 0, 1, 0},
	{0, 0, 0, 0, 1},
	{1, 0, 0, 0, 0},
	{0, 1, 0, 0, 0},
	{0, 0, 0, 1, 0},
	{0, 0, 0, 0, 1},
}

// Y is inputs
var Y = [][]float64{
	{0, 1, 0, 0, 0},
	{0, 0, 0, 1, 0},
	{0, 0, 0, 0, 1},
	{1, 0, 0, 0, 0},
	{0, 1, 0, 0, 0},
	{0, 0, 0, 1, 0},
	{0, 0, 0, 0, 1},
	{0, 0, 0, 0, 0},
}

var wa [][]float64
var wi [][]float64
var wf [][]float64
var wo [][]float64
var ua [][]float64
var ui [][]float64
var uf [][]float64
var uo [][]float64
var ba [][]float64
var bi [][]float64
var bf [][]float64
var bo [][]float64

var trains = 100000
var alpha = 0.1
var layers = 8

func main() {

	alphaW := zerosAlpha(1, 1)
	alphaU := zerosAlpha(1, 1)
	alphaB := zerosAlpha(1, 5)

	initWeight()

	for n := 0; n < trains; n++ {

		outPrevs := zeros(1, 5)
		statePrevs := zeros(1, 5)
		states := zeros(1, 5)
		outs := [][]float64{}
		as := [][]float64{}
		is := [][]float64{}
		fs := [][]float64{}
		os := [][]float64{}

		for i, v := range X {
			x := [][]float64{}
			x = append(x, v)
			outPrevsI := [][]float64{}
			outPrevsI = append(outPrevsI, outPrevs[i])
			a := candidateGate(x, outPrevsI)
			ig := inputGate(x, outPrevsI)
			f := forgetGate(x, outPrevsI)
			o := outputGate(x, outPrevsI)
			// fmt.Println("a: ", a)
			// fmt.Println("i: ", ig)
			// fmt.Println("f: ", f)
			// fmt.Println("o: ", o)

			statePrevsI := [][]float64{}
			statePrevsI = append(statePrevsI, statePrevs[i])
			c1 := darab(a, ig)
			c2 := darab(f, statePrevsI)
			c := tambah(c1, c2)
			out := darab(o, tanH(c))
			// fmt.Println("state: ", c[0])
			// fmt.Println("out: ", out[0])

			outPrevs = append(outPrevs, out[0])
			statePrevs = append(statePrevs, c[0])
			outs = append(outs, out[0])
			as = append(as, a[0])
			is = append(is, ig[0])
			fs = append(fs, f[0])
			os = append(os, o[0])
			states = append(states, c[0])

			// if n%1000 == 0 {
			// 	fmt.Println("input: ", x)
			// 	fmt.Println("output: ", out)
			// }
		}

		triOutAtts := zeros(1, 5)
		BPDeltaStates := zeros(1, 5)
		// Append 0s to future f
		fs = append(fs, BPDeltaStates[0])

		i2 := 0
		// OR
		// newI = layers - i - 1

		dWa := zeros(1, 1)
		dWi := zeros(1, 1)
		dWf := zeros(1, 1)
		dWo := zeros(1, 1)

		dUa := zeros(1, 1)
		dUI := zeros(1, 1)
		dUf := zeros(1, 1)
		dUo := zeros(1, 1)

		dBa := zeros(1, 5)
		dBi := zeros(1, 5)
		dBf := zeros(1, 5)
		dBo := zeros(1, 5)

		for i := layers - 1; i >= 0; i-- {

			out := [][]float64{}
			out = append(out, outs[i])
			y := [][]float64{}
			y = append(y, Y[i])
			// triangle delta @ t
			triAtt := tolak(out, y)
			// fmt.Println("triangle@t: ", triAtt[0])

			// triangle out @ t
			triOutAtt := [][]float64{}
			triOutAtt = append(triOutAtt, triOutAtts[i2])
			// fmt.Println("triangleOut@t: ", triOutAtt[0])
			dOut := tambah(triAtt, triOutAtt)
			// fmt.Println("dOut: ", dOut[0])

			o := [][]float64{}
			o = append(o, os[i])

			state := [][]float64{}
			state = append(state, states[i+1])

			BPDeltaStatesI := [][]float64{}
			BPDeltaStatesI = append(BPDeltaStatesI, BPDeltaStates[i2])

			fFuture := [][]float64{}
			fFuture = append(fFuture, fs[i+1])

			dState := darab(dOut, o)
			dState = darab(dState, tanHDeriv(state))
			dState = tambah(dState, darab(BPDeltaStatesI, fFuture))
			// fmt.Println("dState: ", dState[0])

			ig := [][]float64{}
			ig = append(ig, is[i])

			a := [][]float64{}
			a = append(a, as[i])

			dA := darab(dState, ig)
			dA = darab(dA, oneMinusSquare(a))
			// fmt.Println("dA: ", dA[0])

			dI := darab(dState, a)
			dI = darab(dI, sigmoidDeriv(ig))
			// fmt.Println("dI: ", dI[0])

			statePrev := [][]float64{}
			statePrev = append(statePrev, states[i])

			f := [][]float64{}
			f = append(f, fs[i])

			// fmt.Println("states: ", states)
			// fmt.Println("i-1: ", i-1)
			// fmt.Println("state-1: ", statePrev)
			dF := darab(dState, statePrev)
			dF = darab(dF, sigmoidDeriv(f))
			// fmt.Println("dF: ", dF[0])

			dO := darab(dOut, tanH(state))
			dO = darab(dO, sigmoidDeriv(o))
			// fmt.Println("dO: ", dO[0])

			// cWs := compileWs()
			cUs := compileUs()
			cGs := compileGates(dA, dI, dF, dO)

			// dX is declared but not used (check note why not use)
			// dX := dot(cWs, cGs)
			// fmt.Println("dX: ", dX)

			dOutNext := dot(cUs, cGs)
			// fmt.Println("dOutNext: ", dOutNext)

			BPDeltaStates = append(BPDeltaStates, dState[0])
			triOutAtts = append(triOutAtts, dOutNext[0])

			x := [][]float64{}
			x = append(x, X[i])
			dWa = tambah(dWa, dot(transpose(dA), x))
			dWi = tambah(dWi, dot(transpose(dI), x))
			dWf = tambah(dWf, dot(transpose(dF), x))
			dWo = tambah(dWo, dot(transpose(dO), x))

			if i != 0 {
				outPrev := [][]float64{}
				outPrev = append(outPrev, outs[i-1])

				dUa = tambah(dUa, dot(transpose(dA), outPrev))
				dUI = tambah(dUI, dot(transpose(dI), outPrev))
				dUf = tambah(dUf, dot(transpose(dF), outPrev))
				dUo = tambah(dUo, dot(transpose(dO), outPrev))
			}

			dBa = tambah(dBa, dA)
			dBi = tambah(dBi, dI)
			dBf = tambah(dBf, dF)
			dBo = tambah(dBo, dO)

			i2++
		}

		wa = tolak(wa, darab(alphaW, dWa))
		wi = tolak(wi, darab(alphaW, dWi))
		wf = tolak(wf, darab(alphaW, dWf))
		wo = tolak(wo, darab(alphaW, dWo))

		ua = tolak(ua, darab(alphaU, dUa))
		ui = tolak(ui, darab(alphaU, dUI))
		uf = tolak(uf, darab(alphaU, dUf))
		uo = tolak(uo, darab(alphaU, dUo))

		ba = tolak(ba, darab(alphaB, dBa))
		bi = tolak(bi, darab(alphaB, dBi))
		bf = tolak(bf, darab(alphaB, dBf))
		bo = tolak(bo, darab(alphaB, dBo))

		// fmt.Println("wa: ", wa)
		// fmt.Println("wi: ", wi)
		// fmt.Println("wf: ", wf)
		// fmt.Println("wo: ", wo)
		// fmt.Println("ua: ", ua)
		// fmt.Println("ui: ", ui)
		// fmt.Println("uf: ", uf)
		// fmt.Println("uo: ", uo)
		// fmt.Println("Ba: ", ba)
		// fmt.Println("Bi: ", bi)
		// fmt.Println("Bf: ", bf)
		// fmt.Println("Bo: ", bo)

	}

	fmt.Println("Ready!")
	for i := 0; i < 10; i++ {
		testPower()
	}

}

func testPower() {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	if scanner.Err() != nil {
		log.Fatalln(scanner.Err())
	}
	txt := scanner.Text()
	x := [][]float64{{0, 0, 0, 0, 0}}
	outPrevsI := [][]float64{}
	outPrevsI = append(outPrevsI, x[0])
	statePrevsI := [][]float64{}
	statePrevsI = append(statePrevsI, x[0])
	// there should be 5 vectors here "Jane0 saw1 Doug2 Spot3 .4"
	switch txt {
	case "jane":
		x[0][0] = 1
		x[0][1] = 0
		x[0][2] = 0
		x[0][3] = 0
		x[0][4] = 0
	case "saw":
		x[0][0] = 0
		x[0][1] = 1
		x[0][2] = 0
		x[0][3] = 0
		x[0][4] = 0
	case "doug":
		x[0][0] = 0
		x[0][1] = 0
		x[0][2] = 1
		x[0][3] = 0
		x[0][4] = 0
	case "spot":
		x[0][0] = 0
		x[0][1] = 0
		x[0][2] = 0
		x[0][3] = 1
		x[0][4] = 0
	case ".":
		x[0][0] = 0
		x[0][1] = 0
		x[0][2] = 0
		x[0][3] = 0
		x[0][4] = 1
	default:
		x[0][0] = 0
		x[0][1] = 0
		x[0][2] = 0
		x[0][3] = 0
		x[0][4] = 1
	}

	fmt.Println("txt:", x)

	a := candidateGate(x, outPrevsI)
	ig := inputGate(x, outPrevsI)
	f := forgetGate(x, outPrevsI)
	o := outputGate(x, outPrevsI)

	c1 := darab(a, ig)
	c2 := darab(f, statePrevsI)
	c := tambah(c1, c2)
	out := darab(o, tanH(c))

	fmt.Println("out: ", out)

	max := 0.00
	index := 0

	for _, v := range out {
		for i, v2 := range v {
			if i == 0 {
				max = v2
			} else {
				if max < v2 {
					max = v2
					index = i
				}
			}
			// fmt.Println("txt:", v2)
			// fmt.Println("max:", max)
		}
		// fmt.Println("txt:", txt)
		// fmt.Println("iterate h:", v)
		// fmt.Println("iterate h0:", v[0])
		// fmt.Println("iterate x0:", x[0][0])
	}

	switch index {
	case 0:
		fmt.Println("jane")
	case 1:
		fmt.Println("saw")
	case 2:
		fmt.Println("doug")
	case 3:
		fmt.Println("spot")
	case 4:
		fmt.Println(".")
	default:
		fmt.Println("lol")
	}

}

func candidateGate(x, out [][]float64) [][]float64 {
	hc1 := dot(wa, x)
	hc2 := dot(ua, out)
	hc := tambah(hc1, hc2)
	hc = tambah(hc, ba)

	return tanH(hc)
}

func inputGate(x, out [][]float64) [][]float64 {
	hi1 := dot(wi, x)
	hi2 := dot(ui, out)
	hi := tambah(hi1, hi2)
	hi = tambah(hi, bi)

	return sigmoid(hi)
}

func forgetGate(x, out [][]float64) [][]float64 {
	hf1 := dot(wf, x)
	hf2 := dot(uf, out)
	hf := tambah(hf1, hf2)
	hf = tambah(hf, bf)

	return sigmoid(hf)
}

func outputGate(x, out [][]float64) [][]float64 {
	ho1 := dot(wo, x)
	ho2 := dot(uo, out)
	ho := tambah(ho1, ho2)
	ho = tambah(ho, bo)

	return sigmoid(ho)
}

func compileWs() [][]float64 {
	r := 2
	c := 4
	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			// v[i][j] = 0
			switch j {
			case 0:
				v[i][j] = wa[i][0]
			case 1:
				v[i][j] = wi[i][0]
			case 2:
				v[i][j] = wf[i][0]
			case 3:
				v[i][j] = wo[i][0]
			}
		}
	}

	// fmt.Println("compileWs: ", v)

	return v
}

func compileUs() [][]float64 {
	r := 1
	c := 4
	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			// v[i][j] = 0
			switch j {
			case 0:
				v[i][j] = ua[i][0]
			case 1:
				v[i][j] = ui[i][0]
			case 2:
				v[i][j] = uf[i][0]
			case 3:
				v[i][j] = uo[i][0]
			}
		}
	}

	// fmt.Println("compileUs: ", v)

	return v
}

func compileGates(a, is, f, o [][]float64) [][]float64 {
	r := 4
	c := 5
	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			// v[i][j] = 0
			switch i {
			case 0:
				v[i][j] = a[0][0]
			case 1:
				v[i][j] = is[0][0]
			case 2:
				v[i][j] = f[0][0]
			case 3:
				v[i][j] = o[0][0]
			}
		}
	}

	// fmt.Println("compileGates: ", v)

	return v
}

func randomizeValue(r int, c int) [][]float64 {
	//we are seeding the rand variable with present time
	//so that we would get different output each time
	rand.Seed(time.Now().UnixNano())
	// OR WE CAN JUST CONSTANT IT FOR NOW!
	// rand.Seed(0)

	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v[i][j] = 2*rand.Float64() - 1
		}
	}

	return v
}

func initWeight() {
	wa = randomizeValue(1, 1)
	ua = randomizeValue(1, 1)
	ba = zeros(1, 5)

	// get the dimensions of matrix
	// init neural network weights
	wf = randomizeValue(1, 1)
	uf = randomizeValue(1, 1)
	bf = zeros(1, 5)

	wi = randomizeValue(1, 1)
	ui = randomizeValue(1, 1)
	bi = zeros(1, 5)

	wo = randomizeValue(1, 1)
	uo = randomizeValue(1, 1)
	bo = zeros(1, 5)
}

func zeros(r int, c int) [][]float64 {
	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v[i][j] = 0
		}
	}

	return v
}

func zerosAlpha(r int, c int) [][]float64 {
	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v[i][j] = alpha
		}
	}

	return v
}

func sigmoidDeriv(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			output[i][i2] = v2 * (1 - v2)
		}
	}

	return output
}

func sigmoid(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			var nX float64
			nX = 0 - v2
			output[i][i2] = 1 / (1 + math.Exp(nX))
		}
	}

	return output
}

func tanH(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			output[i][i2] = math.Tanh(v2)
		}
	}

	return output
}

func tanHDeriv(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			output[i][i2] = 1 - math.Pow(math.Tanh(v2), 2)
		}
	}

	return output
}

func oneMinusSquare(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			output[i][i2] = 1 - math.Pow(v2, 2)
		}
	}

	return output
}

func dot(m1 [][]float64, m2 [][]float64) [][]float64 {

	// Ref 2d slice
	// https://gobyexample.com/slices
	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m2[0]))
	}

	// outR := len(m1)
	// outC := len(m2[0])

	for outR, v := range m1 {
		for outC := range m2[0] {
			output[outR][outC] = 0
			for i2, v2 := range v {
				output[outR][outC] += v2 * m2[i2][outC]
			}
		}
	}

	return output
}

func darab(m1 [][]float64, m2 [][]float64) [][]float64 {

	product := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		product[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			product[i][i2] = v2 * m2[i][i2]
		}
	}

	return product
}

func tambah(m1 [][]float64, m2 [][]float64) [][]float64 {

	sum := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		sum[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			sum[i][i2] = v2 + m2[i][i2]
		}
	}

	return sum
}

func tolak(m1 [][]float64, m2 [][]float64) [][]float64 {

	difference := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		difference[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			difference[i][i2] = v2 - m2[i][i2]
		}
	}

	return difference
}

func transpose(m1 [][]float64) [][]float64 {

	mT := make([][]float64, len(m1[0]))
	for i := 0; i < len(m1[0]); i++ {
		mT[i] = make([]float64, len(m1))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			mT[i2][i] = v2
		}
	}

	return mT
}

func softmax(m1 [][]float64) [][]float64 {

	sumZExp := 0.00
	s := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		s[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			s[i][i2] = math.Exp(v2)
		}
	}

	for _, v := range s {
		for _, v2 := range v {
			sumZExp += v2
		}
	}

	for i, v := range s {
		for i2, v2 := range v {
			s[i][i2] = v2 / sumZExp
		}
	}

	return s
}
