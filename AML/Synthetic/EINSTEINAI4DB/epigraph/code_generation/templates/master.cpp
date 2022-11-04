#include <pybind11/pybind11.h>


#include <pybind11/stl.h>
#include <vector>
using namespace std;
namespace py = pybind11;

// This is a simple example of a C++ function that takes and returns a
// vector of doubles.
vector<double> add_one(vector<double> v) {
    for (auto &x : v) {
        x += 1;
    }
    return v;
}




{methods}

// There is a module named 'example' in this example, so use 'm.def' to
// add new functions to the module. The first argument is the name of the
// function in Python, the second is the C++ function that implements it,
// and the third is the "docstring" shown in Python help().

PYBIND11_MODULE(optimized_inference, m){{
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.doc() = "Generated mumford_switch ensemble code";
    {registration}

}}