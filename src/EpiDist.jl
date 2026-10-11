module EpiDist

export Commons, Null, Nash, Ctrl, Adpt, DevC, DevL

include("commons.jl")
using .Commons

module Null
    using ..Commons
    include("null.jl")
end

module Nash
    using ..Commons
    include("nash.jl")
end

module Ctrl
    using ..Commons
    include("controlled.jl")
end

module Adpt
    using ..Commons
    include("adaptive.jl")
end

module DevC
    using ..Commons
    include("deviator_CO.jl")
end

module DevL
    using ..Commons
    include("deviator_LF.jl")
end

end
