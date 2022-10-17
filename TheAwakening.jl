### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ 813be572-4d04-11ed-3a65-17837979498e
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ 33e7917b-58b3-4a92-9c30-2cd9dfa4623b
import LinearAlgebra: kron, det

# ╔═╡ 3ea4ae06-c32b-4386-a75b-3292f2947c06
# ensures that a qubit is valid
verify_magnitude_sum(zs::Number...) = sum((z->abs(z)^2).(zs)) ≈ 1

# ╔═╡ c5246cb2-e0ba-4c15-8af9-c3c088c62d97
begin
	# representations of a qubit and its properties
	struct Qubit
		# |Ψ⟩ = α|0⟩ + β|1⟩
		# |α|² + |β|² = 1
		α::Complex
		β::Complex
		Qubit() = new(1, 0)
		Qubit(α::ComplexF64, β::ComplexF64) = verify_magnitude_sum(α, β) ? new(α, β) : error("Invalid Probability Amplitudes")
		Qubit(θ::Real, ϕ::Real) = new(cos((π*θ/180)/2)+0.0im, exp(im*(π*ϕ/180))*sin((π*θ/180)/2)+0.0im)
		Qubit(v::Matrix{ComplexF64}) = new(v[1:2]...)
	end
	qubit_vector(q::Qubit)::Matrix = convert.(ComplexF64, reshape([q.α, q.β], (2, 1)))
	multi_qubit_vector(qs::Qubit...)::Matrix = kron(qubit_vector.(qs)...)	
	struct NQubit{N}
		# |ψ⟩ = ∑ᵢαᵢ|bin(i)⟩
		# ∑ᵢ|αᵢ|² = 1
		coefficients::Matrix{ComplexF64}
		NQubit(qubits::Qubit...) = length(qubits)>1 ? new{length(qubits)}(multi_qubit_vector(qubits...)) : new{length(qubits)}(qubit_vector(qubits[1]))
		NQubit(qubits::Matrix{ComplexF64}) = new{length(qubits)}(qubits)
	end
	custom_round(ψ::NQubit) = NQubit(custom_round.(ψ.coefficients))
	
	import Base: *
	a::Matrix{ComplexF64} * q::Qubit = Qubit(a * reshape([q.α, q.β], (2, 1)))
	a::Matrix{ComplexF64} * q::NQubit = NQubit(a * q.coefficients)
end

# ╔═╡ 33442535-c871-400b-a3d0-a21f9a570318
# rounding complex numbers
custom_round(z, n_digits=10) = round(real(z), digits=n_digits) + round(imag(z), digits=n_digits)*im

# ╔═╡ 19374d2b-d2be-4e8d-a221-16589c3451f1
begin
	# nice way to see the wave equation
	linear_superposition_representation(ψ::Matrix) = join(["($(custom_round(ψ[i])))|$(lpad(string(i-1, base=2), Int(log2(length(ψ))), "0"))⟩" for i in 1:length(ψ)], " + ")
	linear_superposition_representation(ψ::NQubit) = linear_superposition_representation(ψ.coefficients)
	coefficients_probabilities(ψ::Matrix) = [
		["|$(lpad(string(i-1, base=2), Int(log2(length(ψ))), "0"))⟩" for i in 1:length(ψ)],
		["$(custom_round(ψ[i]))" for i in 1:length(ψ)],
		["$(round(abs2(ψ[i]), digits=10))" for i in 1:length(ψ)]
	]
	coefficients_probabilities(ψ::NQubit) = coefficients_probabilities(ψ.coefficients)
end

# ╔═╡ 964284dc-af05-4a4a-9616-349bf8cea2b6
begin
	# nice way to see collapsed states
	probabilities(ψ::Matrix) = round.(ψ, digits=5), probabilities_only(ψ)
	probabilities(ψ::NQubit) = probabilities(ψ.coefficients)
	probabilities_only(ψ::Matrix) = vcat(["|$(lpad(string(i-1, base=2), Int(log2(length(ψ))), "0"))⟩: $(round(abs(ψ[i])^2, digits=5))" for i in 1:length(ψ)]...)
	probabilities_only(ψ::NQubit) = probabilities_only(ψ.coefficients)
end

# ╔═╡ 8ae7ff31-b6c4-4aee-b163-6b5c8d12bb90
begin
	
	# 1-qubit gates
	R_x(θ::Real) = [cos(θ/2) -im*sin(θ/2); -im*sin(θ/2) cos(θ/2)]
	R_y(θ::Real) = [cos(θ/2) -sin(θ/2); sin(θ/2) cos(θ/2)]
	R_z(θ::Real) = [exp(-im*θ/2) 0; 0 exp(im*θ/2)]
	P(λ) = [1 0; 0 exp(im*λ)]
	I = Matrix{ComplexF64}([1 0; 0 1])
	H = Matrix{ComplexF64}([1/√(2) 1/√(2); 1/√(2) -1/√(2)])
	X = Matrix{ComplexF64}([0 1; 1 0])
	Y = Matrix{ComplexF64}([0 -im; im 0])
	Z = Matrix{ComplexF64}([1 0; 0 -1])
	S = Matrix{ComplexF64}([1 0; 0 im])
	T = Matrix{ComplexF64}([1 0; 0 sqrt(im)])
	decompose(U::Matrix{ComplexF64}) = begin
		α = atan(imag(det(U)),real(det(U)))/2
		V = exp(-im*α)*U
		θ = abs(V[1, 1])≥abs(V[1, 2]) ? 2*acos(abs(V[1, 1])) : 2*asin(abs(V[1, 2]))
		if cos(θ/2) == 0
			λ = atan(imag(V[2, 1]/sin(θ/2)), real(V[2, 1]/sin(θ/2)))
			ϕ = -λ
		elseif sin(θ/2) == 0
			ϕ = atan(imag(V[2, 2]/cos(θ/2)), real(V[2, 2]/cos(θ/2)))
			λ = ϕ
		else
			ϕ = atan(imag(V[2, 2]/cos(θ/2)), real(V[2, 2]/cos(θ/2)))+atan(imag(V[2, 1]/sin(θ/2)), real(V[2, 1]/sin(θ/2)))
			λ = 2*atan(imag(V[2, 2]/cos(θ/2)), real(V[2, 2]/cos(θ/2)))-ϕ
		end
		return (custom_round(α), round(real(θ), digits=10), round(real(ϕ), digits=10), round(real(λ), digits=10))
	end
	U₃(γ, θ, ϕ, λ) = custom_round.(exp(im*γ)*R_z(ϕ)*R_y(θ)*R_z(λ))

	# 2-qubit gates
	CU(U::Matrix{ComplexF64})::Matrix{ComplexF64} = hvcat(
		(2, 2), 
		[1 0; 0 1], 
		[0 0; 0 0], 
		[0 0; 0 0], 
		U
	)
	CX = CU(X)
	CZ = CU(Z)
	CS = CU(S)
	CH = CU(H)
	SWAP = Matrix{ComplexF64}([1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1])

	# 3-qubit gates
	CCU(U::Matrix{ComplexF64})::Matrix{ComplexF64} = hvcat(
		(2, 2),
		[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1],
		[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0],
		[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0],
		CU(U)
	)
	CB(B::Matrix{ComplexF64})::Matrix{ComplexF64} = hvcat(
		(2, 2),
		[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1],
		[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0],
		[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0],
		B
	)
	CCX = CCU(X)
	CSWAP = CB(SWAP)

	# N-qubit gates
	QFT(N::Int) = custom_round.([exp(2*i*j*π*im/(2^N))/sqrt(2^N) for i in 0:2^N-1, j in 0:2^N-1])
	IQFT(N::Int) = adjoint(QFT(N))
	C(Us::Matrix{ComplexF64}...) = kron(Us...)
	
end

# ╔═╡ b99350cb-236e-4a1e-a9e6-5d581d76fc65
begin
	Edna = Qubit(1.0+0.0im, 0.0+0.0im)
	Edna = H * Edna
	Edna = U₃(0, -π/6, 0, 0) * Edna
	Edna = U₃(0, -π/6, 0, 0) * Edna
	Edna = U₃(0, -π/6, 0, 0) * Edna
	Robert = Qubit(0.0+0.0im, 1.0+0.0im)
	Edna = H * Edna
	ψ = NQubit(Edna, Robert)
	ψ = CX * ψ
	ψ = C(I, I) * ψ
	ψ |> probabilities_only
end

# ╔═╡ bb28213f-958f-477d-b9c0-fdd31b0606ec
md"""
$\text{Edna} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
$\text{Edna} = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix} * \text{Edna}$
$\text{Edna} = \begin{bmatrix} 0.96526 & 0.258819 \\ -0.258819 & 0.965926 \end{bmatrix} * \text{Edna}$
$\text{Edna} = \begin{bmatrix} 0.96526 & 0.258819 \\ -0.258819 & 0.965926 \end{bmatrix} * \text{Edna}$
$\text{Edna} = \begin{bmatrix} 0.96526 & 0.258819 \\ -0.258819 & 0.965926 \end{bmatrix} * \text{Edna}$
$\text{Robert} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$
$\text{Edna} = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix} * \text{Edna}$
$\text{Entangled Pair} = \begin{bmatrix} 0 \\ \frac{1}{\sqrt{2}} \\ 0 \\ \frac{1}{\sqrt{2}} \end{bmatrix}$
$\text{Entangled Pair} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix} * \text{Entangled Pair}$
$\text{Entangled Pair} = \begin{bmatrix} 0 \\ \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \\ 0 \end{bmatrix}$
$E := \text{Edna's happiness}$
$R := \text{Robert's 'correct' behavior}$
$P((E=0)|(R=0)) = 0^2 = 0 ⟹ \text{whenever Robert acts immorally, Edna is never unhappy}$
$P((E=1)|(R=1)) = 0^2 = 0 ⟹ \text{whenever Robert acts morally, Edna is never happy}$
$P((E=1)|(R=0)) = (\frac{1}{\sqrt{2}})^2 = \frac{1}{2} ⟹ \text{whenever Robert acts immorally, Edna is happy}$
$P((E=0)|(R=1)) = (\frac{1}{\sqrt{2}})^2 = \frac{1}{2} ⟹ \text{whenever Robert acts morally, Edna is unhappy}$
$\text{}$
$\text{Since we know that Robert acted as he thought moral, the system was measured such that R=1.}$
$\text{When R=1, the only possibility for E is E=0, and so Edna was forced to be unhappy and commit suicide.}$
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0-rc3"
manifest_format = "2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
"""

# ╔═╡ Cell order:
# ╟─813be572-4d04-11ed-3a65-17837979498e
# ╠═33e7917b-58b3-4a92-9c30-2cd9dfa4623b
# ╠═3ea4ae06-c32b-4386-a75b-3292f2947c06
# ╠═c5246cb2-e0ba-4c15-8af9-c3c088c62d97
# ╠═33442535-c871-400b-a3d0-a21f9a570318
# ╠═19374d2b-d2be-4e8d-a221-16589c3451f1
# ╠═964284dc-af05-4a4a-9616-349bf8cea2b6
# ╠═8ae7ff31-b6c4-4aee-b163-6b5c8d12bb90
# ╠═b99350cb-236e-4a1e-a9e6-5d581d76fc65
# ╟─bb28213f-958f-477d-b9c0-fdd31b0606ec
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
