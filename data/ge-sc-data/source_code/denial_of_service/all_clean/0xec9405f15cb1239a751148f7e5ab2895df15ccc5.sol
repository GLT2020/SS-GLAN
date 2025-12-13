pragma solidity ^0.4.24;
contract Ownable {
  address private _owner;
  event OwnershipTransferred(
    address indexed previousOwner,
    address indexed newOwner
  );
  constructor() internal {
    _owner = msg.sender;
    emit OwnershipTransferred(address(0), _owner);
  }
  function owner() public view returns(address) {
    return _owner;
  }
  modifier onlyOwner() {
    require(isOwner());
    _;
  }
  function isOwner() public view returns(bool) {
    return msg.sender == _owner;
  }
  function renounceOwnership() public onlyOwner {
    emit OwnershipTransferred(_owner, address(0));
    _owner = address(0);
  }
  function transferOwnership(address newOwner) public onlyOwner {
    _transferOwnership(newOwner);
  }
  function _transferOwnership(address newOwner) internal {
    require(newOwner != address(0));
    emit OwnershipTransferred(_owner, newOwner);
    _owner = newOwner;
  }
}
library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
      return 0;
    }
    uint256 c = a * b;
    require(c / a == b);
    return c;
  }
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b > 0); 
    uint256 c = a / b;
    return c;
  }
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b <= a);
    uint256 c = a - b;
    return c;
  }
  function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    require(c >= a);
    return c;
  }
  function mod(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b != 0);
    return a % b;
  }
}
pragma solidity 0.4.24;
contract Storage is Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private balances;
    mapping (address => mapping (address => uint256)) private allowed;
    uint256 private totalSupply;
    address private _implementor;
    event StorageImplementorTransferred(address indexed from,
                                        address indexed to);
    constructor(address owner, address implementor) public {
        require(
            owner != address(0),
            "Owner should not be the zero address"
        );
        require(
            implementor != address(0),
            "Implementor should not be the zero address"
        );
        transferOwnership(owner);
        _implementor = implementor;
    }
    function isImplementor() public view returns(bool) {
        return msg.sender == _implementor;
    }
    function setBalance(address owner,
                        uint256 value)
        public
        onlyImplementor
    {
        balances[owner] = value;
    }
    function increaseBalance(address owner, uint256 addedValue)
        public
        onlyImplementor
    {
        balances[owner] = balances[owner].add(addedValue);
    }
    function decreaseBalance(address owner, uint256 subtractedValue)
        public
        onlyImplementor
    {
        balances[owner] = balances[owner].sub(subtractedValue);
    }
    function getBalance(address owner)
        public
        view
        returns (uint256)
    {
        return balances[owner];
    }
    function setAllowed(address owner,
                        address spender,
                        uint256 value)
        public
        onlyImplementor
    {
        allowed[owner][spender] = value;
    }
    function increaseAllowed(
        address owner,
        address spender,
        uint256 addedValue
    )
        public
        onlyImplementor
    {
        allowed[owner][spender] = allowed[owner][spender].add(addedValue);
    }
    function decreaseAllowed(
        address owner,
        address spender,
        uint256 subtractedValue
    )
        public
        onlyImplementor
    {
        allowed[owner][spender] = allowed[owner][spender].sub(subtractedValue);
    }
    function getAllowed(address owner,
                        address spender)
        public
        view
        returns (uint256)
    {
        return allowed[owner][spender];
    }
    function setTotalSupply(uint256 value)
        public
        onlyImplementor
    {
        totalSupply = value;
    }
    function getTotalSupply()
        public
        view
        returns (uint256)
    {
        return totalSupply;
    }
    function transferImplementor(address newImplementor)
        public
        requireNonZero(newImplementor)
        onlyImplementorOrOwner
    {
        require(newImplementor != _implementor,
                "Cannot transfer to same implementor as existing");
        address curImplementor = _implementor;
        _implementor = newImplementor;
        emit StorageImplementorTransferred(curImplementor, newImplementor);
    }
    modifier onlyImplementorOrOwner() {
        require(isImplementor() || isOwner(), "Is not implementor or owner");
        _;
    }
    modifier onlyImplementor() {
        require(isImplementor(), "Is not implementor");
        _;
    }
    modifier requireNonZero(address addr) {
        require(addr != address(0), "Expected a non-zero address");
        _;
    }
}