pragma solidity >=0.4.22 <0.6.0;
library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256 c) {
    if (a == 0) {
        return 0;
    }
    c = a * b;
    assert(c / a == b);
    return c;
  }
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    return a / b;
  }
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }
  function add(uint256 a, uint256 b) internal pure returns (uint256 c) {
    c = a + b;
    assert(c >= a);
    return c;
  }
}
contract Ownable {
  function bug_unchk_send21() payable public{
      msg.sender.transfer(1 ether);}
  address public owner;
    constructor() public {
        owner = msg.sender;
    }
function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
}
contract TokenERC20 is Ownable {
    using SafeMath for uint256;
  function bug_unchk_send10() payable public{
      msg.sender.transfer(1 ether);}
  string public name;
  function bug_unchk_send22() payable public{
      msg.sender.transfer(1 ether);}
  string public symbol;
  function bug_unchk_send12() payable public{
      msg.sender.transfer(1 ether);}
  uint8 public decimals;
  function bug_unchk_send11() payable public{
      msg.sender.transfer(1 ether);}
  uint256 private _totalSupply;
  function bug_unchk_send1() payable public{
      msg.sender.transfer(1 ether);}
  uint256 public cap;
  function bug_unchk_send2() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => uint256) private _balances;
  function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => mapping (address => uint256)) private _allowed;
  function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
  event Transfer(address indexed from, address indexed to, uint256 value);
  function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
  event Approval(address indexed owner, address indexed spender, uint256 value);
  function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
  event Mint(address indexed to, uint256 amount);
    modifier onlyPayloadSize(uint size) {
        require(msg.data.length >= size + 4);
        _;
    }
    constructor(
        uint256 _cap,
        uint256 _initialSupply,
        string memory _name,
        string memory _symbol,
        uint8 _decimals
    ) public {
        require(_cap >= _initialSupply);
        cap = _cap;
        name = _name;                                       
        symbol = _symbol;                                   
        decimals = _decimals;                               
        _totalSupply = _initialSupply;                      
        _balances[owner] = _totalSupply;                    
        emit Transfer(address(0), owner, _totalSupply);
    }
function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }
function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
    function balanceOf(address _owner) public view returns (uint256) {
        return _balances[_owner];
    }
function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
    function allowance(address _owner, address _spender) public view returns (uint256) {
        return _allowed[_owner][_spender];
    }
function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address _to, uint256 _value) onlyPayloadSize(2 * 32) public returns (bool) {
        _transfer(msg.sender, _to, _value);
        return true;
    }
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    function approve(address _spender, uint256 _value) public returns (bool) {
        _approve(msg.sender, _spender, _value);
        return true;
    }
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    function transferFrom(address _from, address _to, uint256 _value) onlyPayloadSize(3 * 32) public returns (bool) {
        _transfer(_from, _to, _value);
        _approve(_from, msg.sender, _allowed[_from][msg.sender].sub(_value));
        return true;
    }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
    function _transfer(address _from, address _to, uint256 _value) internal {
        require(_to != address(0), "ERC20: transfer to the zero address");
        _balances[_from] = _balances[_from].sub(_value);
        _balances[_to] = _balances[_to].add(_value);
        emit Transfer(_from, _to, _value);
    }
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
    function _approve(address _owner, address _spender, uint256 _value) internal {
        require(_owner != address(0), "ERC20: approve from the zero address");
        require(_spender != address(0), "ERC20: approve to the zero address");
        _allowed[_owner][_spender] = _value;
        emit Approval(_owner, _spender, _value);
    }
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
    function mint(address _to, uint256 _amount) onlyOwner public returns (bool) {
        require(_totalSupply.add(_amount) <= cap);
        _totalSupply = _totalSupply.add(_amount);
        _balances[_to] = _balances[_to].add(_amount);
        emit Mint(_to, _amount);
        emit Transfer(address(0), _to, _amount);
        return true;
    }
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
    function transferBatch(address[] memory _tos, uint256[] memory _values) public returns (bool) {
        require(_tos.length == _values.length);
        for (uint256 i = 0; i < _tos.length; i++) {
            transfer(_tos[i], _values[i]);
        }
        return true;
    }
function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
}
contract XLToken is TokenERC20 {
    constructor() TokenERC20(18*10**16, 12*10**16, "XL Token", "XL", 8) public {}
function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
}