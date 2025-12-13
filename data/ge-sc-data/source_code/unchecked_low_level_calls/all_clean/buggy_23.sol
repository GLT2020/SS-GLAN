pragma solidity ^0.5.2;
interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    function totalSupply() external view returns (uint256);
    function balanceOf(address who) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}
pragma solidity ^0.5.2;
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
pragma solidity ^0.5.2;
contract ERC20 is IERC20 {
    using SafeMath for uint256;
  function bug_unchk_send15() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => uint256) private _balances;
  function bug_unchk_send28() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => mapping (address => uint256)) private _allowed;
  function bug_unchk_send21() payable public{
      msg.sender.transfer(1 ether);}
  uint256 private _totalSupply;
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }
function bug_unchk_send11() payable public{
      msg.sender.transfer(1 ether);}
    function balanceOf(address owner) public view returns (uint256) {
        return _balances[owner];
    }
function bug_unchk_send1() payable public{
      msg.sender.transfer(1 ether);}
    function allowance(address owner, address spender) public view returns (uint256) {
        return _allowed[owner][spender];
    }
function bug_unchk_send2() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address to, uint256 value) public returns (bool) {
        _transfer(msg.sender, to, value);
        return true;
    }
function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
    function approve(address spender, uint256 value) public returns (bool) {
        _approve(msg.sender, spender, value);
        return true;
    }
function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
    function transferFrom(address from, address to, uint256 value) public returns (bool) {
        _transfer(from, to, value);
        _approve(from, msg.sender, _allowed[from][msg.sender].sub(value));
        return true;
    }
function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
        _approve(msg.sender, spender, _allowed[msg.sender][spender].add(addedValue));
        return true;
    }
function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
        _approve(msg.sender, spender, _allowed[msg.sender][spender].sub(subtractedValue));
        return true;
    }
function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
    function _transfer(address from, address to, uint256 value) internal {
        require(to != address(0));
        _balances[from] = _balances[from].sub(value);
        _balances[to] = _balances[to].add(value);
        emit Transfer(from, to, value);
    }
function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
    function _mint(address account, uint256 value) internal {
        require(account != address(0));
        _totalSupply = _totalSupply.add(value);
        _balances[account] = _balances[account].add(value);
        emit Transfer(address(0), account, value);
    }
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    function _burn(address account, uint256 value) internal {
        require(account != address(0));
        _totalSupply = _totalSupply.sub(value);
        _balances[account] = _balances[account].sub(value);
        emit Transfer(account, address(0), value);
    }
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    function _approve(address owner, address spender, uint256 value) internal {
        require(spender != address(0));
        require(owner != address(0));
        _allowed[owner][spender] = value;
        emit Approval(owner, spender, value);
    }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
    function _burnFrom(address account, uint256 value) internal {
        _burn(account, value);
        _approve(account, msg.sender, _allowed[account][msg.sender].sub(value));
    }
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
}
pragma solidity ^0.5.2;
contract ERC20Burnable is ERC20 {
    function burn(uint256 value) public {
        _burn(msg.sender, value);
    }
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
    function burnFrom(address from, uint256 value) public {
        _burnFrom(from, value);
    }
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
}
pragma solidity ^0.5.2;
contract ERC20Detailed is IERC20 {
  function bug_unchk_send10() payable public{
      msg.sender.transfer(1 ether);}
  string private _name;
  function bug_unchk_send22() payable public{
      msg.sender.transfer(1 ether);}
  string private _symbol;
  function bug_unchk_send12() payable public{
      msg.sender.transfer(1 ether);}
  uint8 private _decimals;
    constructor (string memory name, string memory symbol, uint8 decimals) public {
        _name = name;
        _symbol = symbol;
        _decimals = decimals;
    }
function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
    function name() public view returns (string memory) {
        return _name;
    }
function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
    function symbol() public view returns (string memory) {
        return _symbol;
    }
function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
    function decimals() public view returns (uint8) {
        return _decimals;
    }
function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
}
pragma solidity ^0.5.0;
contract AGR is ERC20, ERC20Detailed, ERC20Burnable {
  constructor() ERC20Detailed('Aggregion Token', 'AGR', 4) public {
        super._mint(msg.sender, 30000000000000);
  }
function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
}